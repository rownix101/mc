//! 世界存档: 磁盘布局与手写二进制格式.
//!
//! 每个世界占据 `~/.mc/saves/` 下自己的目录 (可用 `MC_SAVE_DIR` 环境变量
//! 覆盖根目录，测试与便携安装可用):
//!
//! ```text
//! .mc/saves/<世界名>/world.dat
//! ```
//!
//! `world.dat` 内容: 魔数 + 格式版本 + 方块注册表版本, 世界名 / 种子 /
//! 出生点, 玩家最后状态 (位置 / 视角 / 模式 / 选中槽 / 非空物品堆), 以及
//! 全部方块编辑 (含 Air)。写入先落临时文件再原子改名, 避免写一半损坏
//! 存档; 读取做严格校验, 任何字段非法都视为存档不可用并跳过。
//!
//! 格式是紧凑小端二进制而不是 JSON: 编辑数量可能上万, 存档走冷路径但
//! 体积敏感, 且零额外依赖.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use glam::DVec3;

use super::block::{Block, BlockId, COUNT as BLOCK_COUNT};
use super::column::{Y_MAX, Y_MIN};
use crate::player::mode::GameMode;

/// 存档根目录下的子目录名.
pub const SAVE_DIR_NAME: &str = "saves";
/// 每个世界目录里的存档文件名.
const FILE_NAME: &str = "world.dat";
/// 文件魔数, 用于区分损坏 / 不相关的文件.
const MAGIC: [u8; 4] = *b"MCS1";
/// 存档格式版本; 不兼容的旧版本整体拒读.
const FORMAT_VERSION: u16 = 1;
/// 世界名长度上限 (UTF-8 字节).
const MAX_NAME_LEN: usize = 128;
/// 单世界编辑数量上限, 防止异常文件拖垮加载.
const MAX_EDITS: usize = 1_000_000;
/// 玩家位置横向坐标上限 (± 十亿格, 远超任何可达世界).
const MAX_POSITION: f64 = 1.0e9;
/// 物品堆数量上限 (MC 标准满堆).
const MAX_STACK_COUNT: u16 = 64;

/// 存档里的单个非空物品堆.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavedStack {
    /// 槽位下标 (0..INVENTORY_SIZE).
    pub slot: usize,
    pub block: Block,
    pub count: u16,
}

/// 退出世界时保存的玩家状态, 重新进入时用于"继续游戏".
#[derive(Clone, Debug, PartialEq)]
pub struct SavedPlayer {
    /// 脚底中心位置.
    pub position: DVec3,
    pub yaw: f64,
    pub pitch: f64,
    pub mode: GameMode,
    /// 快捷栏选中槽 (0..INVENTORY_SIZE).
    pub selected: usize,
    /// 非空物品堆; 空槽在恢复时按 `None` 处理.
    pub stacks: Vec<SavedStack>,
}

/// 一次方块编辑: 位置 + 新方块 (Air 也算, 用于挖空).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockEdit {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub block: Block,
}

/// 一个世界的全部持久化数据.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldSave {
    pub name: String,
    pub seed: u64,
    pub spawn_x: i64,
    pub spawn_z: i64,
    pub spawn_height: i64,
    /// 玩家状态; 刚创建、还没进过游戏的世界为 `None`.
    pub player: Option<SavedPlayer>,
    /// 玩家对生成方块的编辑 (挖掉 / 放置).
    pub edits: Vec<BlockEdit>,
}

impl WorldSave {
    /// 把存档写入 `dir` (不存在则创建)。先写临时文件再原子改名.
    pub fn save(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let bytes = self.encode();
        let tmp = dir.join(format!(".{FILE_NAME}.tmp-{}", std::process::id()));
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, dir.join(FILE_NAME))?;
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(&MAGIC);
        w.u16(FORMAT_VERSION);
        w.u16(BLOCK_COUNT as u16);
        let name = self.name.as_bytes();
        let name_len = name.len().min(MAX_NAME_LEN);
        w.u16(name_len as u16);
        w.bytes(&name[..name_len]);
        w.u64(self.seed);
        w.i64(self.spawn_x);
        w.i64(self.spawn_z);
        w.i64(self.spawn_height);
        match &self.player {
            None => w.u8(0),
            Some(player) => {
                w.u8(1);
                w.f64(player.position.x);
                w.f64(player.position.y);
                w.f64(player.position.z);
                w.f64(player.yaw);
                w.f64(player.pitch);
                w.u8(player.mode.as_u8());
                w.u8(player.selected.min(255) as u8);
                w.u16(player.stacks.len().min(u16::MAX as usize) as u16);
                for stack in &player.stacks {
                    w.u8(stack.slot.min(255) as u8);
                    w.u8(stack.block.id() as u8);
                    w.u16(stack.count);
                }
            }
        }
        w.u32(self.edits.len().min(u32::MAX as usize) as u32);
        for edit in &self.edits {
            w.i64(edit.x);
            w.i64(edit.y);
            w.i64(edit.z);
            w.u8(edit.block.id() as u8);
        }
        w.into_bytes()
    }
}

/// 存档根目录: 优先 `MC_SAVE_DIR`, 否则 `~/.mc/saves`.
///
/// 返回 `None` 表示没有可用位置 (例如找不到用户主目录); 调用方应优雅
/// 降级为纯内存模式, 而不是报错.
pub fn save_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_SAVE_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".mc").join(SAVE_DIR_NAME))
}

/// 世界名到目录名的安全映射: 去掉路径分隔符 / 冒号 / 控制字符,
/// 空白折叠为下划线; 空名字退化为 `world`.
pub fn sanitize_dir_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut pending_space = false;
    for ch in name.chars() {
        match ch {
            '/' | '\\' | ':' | '\0' if !out.is_empty() => pending_space = true,
            '/' | '\\' | ':' | '\0' => {}
            c if (c as u32) < 0x20 => {}
            ' ' if !out.is_empty() => pending_space = true,
            ' ' => {}
            c => {
                if pending_space {
                    out.push('_');
                    pending_space = false;
                }
                out.push(c);
            }
        }
    }
    let out = out.trim_end_matches('_').to_string();
    if out.is_empty() {
        "world".to_string()
    } else {
        out
    }
}

/// 读取 `dir` 里的世界存档; 文件缺失或损坏返回 `None`.
pub fn load_world(dir: &Path) -> Option<WorldSave> {
    let data = fs::read(dir.join(FILE_NAME)).ok()?;
    parse_world_save(&data)
}

/// 列出存档根目录下全部可读世界 (按名字排序); 损坏的目录静默跳过.
pub fn list_worlds(root: &Path) -> Vec<WorldSave> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut worlds = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            worlds.extend(load_world(&path));
        }
    }
    worlds.sort_by(|a, b| a.name.cmp(&b.name));
    worlds
}

/// 为新世界挑选不冲突的目录名: 同名时依次追加 ` (2)`, ` (3)`.
pub fn unique_world_dir(root: &Path, name: &str) -> PathBuf {
    let base = sanitize_dir_name(name);
    let first = root.join(&base);
    if !first.exists() {
        return first;
    }
    let mut suffix = 2u32;
    loop {
        let candidate = root.join(format!("{base} ({suffix})"));
        if !candidate.exists() {
            return candidate;
        }
        suffix += 1;
    }
}

/// 删除整个世界目录 (连同 `world.dat`).
pub fn delete_world(dir: &Path) -> io::Result<()> {
    fs::remove_dir_all(dir)
}

fn parse_world_save(data: &[u8]) -> Option<WorldSave> {
    let mut reader = Reader { data, pos: 0 };
    if !reader
        .take(4)
        .is_some_and(|magic| magic == MAGIC.as_slice())
    {
        return None;
    }
    if reader.u16()? != FORMAT_VERSION {
        return None;
    }
    if reader.u16()? as usize != BLOCK_COUNT {
        return None;
    }
    let name = utf8_field(&mut reader)?;
    if name.is_empty() || name.len() > MAX_NAME_LEN {
        return None;
    }
    let seed = reader.u64()?;
    let (spawn_x, spawn_z, spawn_height) = (
        bounded_i64(reader.i64()?, MAX_POSITION)?,
        bounded_i64(reader.i64()?, MAX_POSITION)?,
        bounded_i64(reader.i64()?, MAX_POSITION)?,
    );
    if !((Y_MIN - 8)..=(Y_MAX + 8)).contains(&spawn_height) {
        return None;
    }
    let player = match reader.u8()? {
        0 => None,
        1 => Some(parse_player(&mut reader)?),
        _ => return None,
    };
    let edit_count = reader.u32()?;
    if edit_count as usize > MAX_EDITS {
        return None;
    }
    let mut edits: Vec<BlockEdit> = Vec::with_capacity(edit_count as usize);
    let mut seen: HashSet<(i64, i64, i64)> = HashSet::new();
    for _ in 0..edit_count {
        let x = bounded_i64(reader.i64()?, MAX_POSITION)?;
        let y = bounded_i64(reader.i64()?, MAX_POSITION)?;
        let z = bounded_i64(reader.i64()?, MAX_POSITION)?;
        if !((Y_MIN - 8)..=(Y_MAX + 8)).contains(&y) {
            return None;
        }
        let block = Block::from_id(reader.u8()? as BlockId)?;
        // 同一位置出现多次编辑时保留最后一条, 与内存中 HashMap 覆盖语义一致.
        let position = (x, y, z);
        if seen.remove(&position) {
            edits.retain(|edit| (edit.x, edit.y, edit.z) != position);
        }
        seen.insert(position);
        edits.push(BlockEdit { x, y, z, block });
    }
    if reader.pos != reader.data.len() {
        return None;
    }
    Some(WorldSave {
        name,
        seed,
        spawn_x,
        spawn_z,
        spawn_height,
        player,
        edits,
    })
}

fn parse_player(reader: &mut Reader) -> Option<SavedPlayer> {
    const SLOT_COUNT: usize = crate::player::inventory::INVENTORY_SIZE;

    let x = bounded_f64(reader.f64()?, MAX_POSITION)?;
    let y = bounded_f64(reader.f64()?, 1.0e6)?;
    let z = bounded_f64(reader.f64()?, MAX_POSITION)?;
    let yaw = bounded_f64(reader.f64()?, std::f64::consts::TAU)?;
    let pitch = reader.f64()?.clamp(-1.55, 1.55);
    if !pitch.is_finite() || !yaw.is_finite() {
        return None;
    }
    if !((Y_MIN - 8) as f64..=(Y_MAX + 8) as f64).contains(&y) {
        return None;
    }
    let mode = match reader.u8()? {
        0 => GameMode::Survival,
        1 => GameMode::Creative,
        _ => return None,
    };
    let selected = reader.u8()? as usize;
    if selected >= SLOT_COUNT {
        return None;
    }
    let stack_count = reader.u16()?;
    if stack_count as usize > SLOT_COUNT {
        return None;
    }
    let mut slots: HashSet<usize> = HashSet::new();
    let mut stacks = Vec::with_capacity(stack_count as usize);
    for _ in 0..stack_count {
        let slot = reader.u8()? as usize;
        if slot >= SLOT_COUNT || !slots.insert(slot) {
            return None;
        }
        let block = Block::from_id(reader.u8()? as BlockId)?;
        let count = reader.u16()?;
        if !(1..=MAX_STACK_COUNT).contains(&count) {
            return None;
        }
        stacks.push(SavedStack { slot, block, count });
    }
    Some(SavedPlayer {
        position: DVec3::new(x, y, z),
        yaw,
        pitch,
        mode,
        selected,
        stacks,
    })
}

fn utf8_field(reader: &mut Reader) -> Option<String> {
    let len = reader.u16()? as usize;
    let bytes = reader.take(len)?;
    let text = std::str::from_utf8(bytes).ok()?;
    Some(text.to_string())
}

fn bounded_i64(value: i64, limit: f64) -> Option<i64> {
    (value.unsigned_abs() as f64 <= limit).then_some(value)
}

fn bounded_f64(value: f64, limit: f64) -> Option<f64> {
    (value.is_finite() && value.abs() <= limit).then_some(value)
}

/// 小端写入器.
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(256),
        }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// 带严格边界检查的小端读取器; 任何越界都返回 `None`.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        let chunk = &self.data[self.pos..end];
        self.pos = end;
        Some(chunk)
    }

    fn u8(&mut self) -> Option<u8> {
        self.take(1)?.first().copied()
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(u64::from_le_bytes(
            self.take(8)?.try_into().ok()?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::player::inventory::{INVENTORY_SIZE, ItemStack};
    use glam::DVec3;

    fn temp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let index = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "mc-save-test-{}-{}-{index}",
            std::process::id(),
            tag
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("创建测试目录");
        dir
    }

    fn sample_save() -> WorldSave {
        WorldSave {
            name: "测试 世界/名".to_string(),
            seed: 0xDEAD_BEEF_CAFE_F00D,
            spawn_x: 3,
            spawn_z: -4,
            spawn_height: 63,
            player: Some(SavedPlayer {
                position: DVec3::new(3.5, 64.0, -3.5),
                yaw: 0.785,
                pitch: -0.2,
                mode: GameMode::Creative,
                selected: INVENTORY_SIZE - HOTBAR_INDEX,
                stacks: vec![
                    SavedStack {
                        slot: INVENTORY_SIZE - HOTBAR_SIZE,
                        block: Block::GrassBlock,
                        count: 64,
                    },
                    SavedStack {
                        slot: INVENTORY_SIZE - HOTBAR_SIZE + 1,
                        block: Block::DiamondOre,
                        count: 1,
                    },
                ],
            }),
            edits: vec![
                BlockEdit {
                    x: 10,
                    y: 61,
                    z: 10,
                    block: Block::Air,
                },
                BlockEdit {
                    x: 11,
                    y: 62,
                    z: 10,
                    block: Block::OakLog,
                },
            ],
        }
    }

    const HOTBAR_SIZE: usize = 9;
    const HOTBAR_INDEX: usize = 8;

    #[test]
    fn roundtrip_full_save() {
        let dir = temp_dir("roundtrip");
        let original = sample_save();
        original.save(&dir).expect("写入存档");
        let loaded = load_world(&dir).expect("应当能读回存档");
        assert_eq!(loaded, original);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_without_player_roundtrips() {
        let dir = temp_dir("no-player");
        let original = WorldSave {
            player: None,
            edits: Vec::new(),
            ..sample_save()
        };
        original.save(&dir).expect("写入存档");
        let loaded = load_world(&dir).expect("应当能读回存档");
        assert_eq!(loaded, original);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_file_is_rejected() {
        let dir = temp_dir("truncated");
        sample_save().save(&dir).expect("写入存档");
        let data = fs::read(dir.join(FILE_NAME)).expect("读取文件");
        fs::write(dir.join(FILE_NAME), &data[..data.len() - 3]).expect("截断文件");
        assert!(load_world(&dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let dir = temp_dir("magic");
        sample_save().save(&dir).expect("写入存档");
        let mut data = fs::read(dir.join(FILE_NAME)).expect("读取文件");
        data[0] = b'X';
        fs::write(dir.join(FILE_NAME), &data).expect("破坏魔数");
        assert!(load_world(&dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_block_id_is_rejected() {
        let dir = temp_dir("block-id");
        // 直接构造一个带非法方块 id 的文件.
        let mut writer = Writer::new();
        writer.bytes(&MAGIC);
        writer.u16(FORMAT_VERSION);
        writer.u16(BLOCK_COUNT as u16);
        writer.u16(1);
        writer.bytes(b"a");
        writer.u64(1);
        writer.i64(0);
        writer.i64(0);
        writer.i64(0);
        writer.u8(0); // no player
        writer.u32(1);
        writer.i64(1);
        writer.i64(1);
        writer.i64(1);
        writer.u8(0xFF); // 非法方块 id
        fs::write(dir.join(FILE_NAME), writer.into_bytes()).expect("写入文件");
        assert!(load_world(&dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_stack_slot_is_rejected() {
        let dir = temp_dir("duplicate-stack-slot");
        // 构造重复槽位文件: 两个 stack 共用 slot 0.
        let mut writer = Writer::new();
        writer.bytes(&MAGIC);
        writer.u16(FORMAT_VERSION);
        writer.u16(BLOCK_COUNT as u16);
        writer.u16(1);
        writer.bytes(b"a");
        writer.u64(1);
        writer.i64(0);
        writer.i64(0);
        writer.i64(0);
        writer.u8(1);
        writer.f64(1.0);
        writer.f64(1.0);
        writer.f64(1.0);
        writer.f64(0.0);
        writer.f64(0.0);
        writer.u8(0);
        writer.u8(0);
        writer.u16(2);
        for _ in 0..2 {
            writer.u8(0); // 同一个槽位写两次
            writer.u8(Block::Stone.id() as u8);
            writer.u16(1);
        }
        writer.u32(0);
        fs::write(dir.join(FILE_NAME), writer.into_bytes()).expect("写入文件");
        assert!(load_world(&dir).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_edits_keep_last() {
        let mut save = sample_save();
        save.edits.push(BlockEdit {
            x: 10,
            y: 61,
            z: 10,
            block: Block::Stone,
        });
        let dir = temp_dir("dup-edit");
        save.save(&dir).expect("写入存档");
        let loaded = load_world(&dir).expect("应当能读回存档");
        let at_position: Vec<Block> = loaded
            .edits
            .iter()
            .filter(|edit| (edit.x, edit.y, edit.z) == (10, 61, 10))
            .map(|edit| edit.block)
            .collect();
        assert_eq!(at_position, vec![Block::Stone]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_name_strips_path_characters() {
        assert_eq!(sanitize_dir_name("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize_dir_name("  x  "), "x");
        assert_eq!(sanitize_dir_name("///"), "world");
        assert_eq!(sanitize_dir_name("新世界"), "新世界");
    }

    #[test]
    fn unique_dir_avoids_collision() {
        let root = temp_dir("unique");
        fs::create_dir_all(&root).expect("创建根目录");
        let first = unique_world_dir(&root, "a");
        fs::create_dir_all(&first).expect("创建第一个目录");
        let second = unique_world_dir(&root, "a");
        assert_ne!(first, second);
        assert!(second.ends_with("a (2)"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn list_worlds_skips_invalid_entries() {
        let root = temp_dir("list");
        let good = root.join("good");
        let mut save = sample_save();
        save.name = "good".to_string();
        save.save(&good).expect("写入好存档");
        fs::create_dir_all(root.join("empty-dir")).expect("创建空目录");
        fs::write(root.join("not-a-dir"), b"junk").expect("写入垃圾文件");
        let bad = root.join("corrupt");
        fs::create_dir_all(&bad).expect("创建坏目录");
        fs::write(bad.join(FILE_NAME), b"not a save").expect("写入坏文件");

        let worlds = list_worlds(&root);
        assert_eq!(worlds.len(), 1);
        assert_eq!(worlds[0].name, "good");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_world_removes_directory() {
        let dir = temp_dir("delete");
        sample_save().save(&dir).expect("写入存档");
        delete_world(&dir).expect("删除世界");
        assert!(!dir.exists());
    }

    #[test]
    fn restored_inventory_slots_match_stacks() {
        let original = sample_save();
        let mut slots = [None; INVENTORY_SIZE];
        if let Some(player) = &original.player {
            for stack in &player.stacks {
                slots[stack.slot] = Some(ItemStack::new(stack.block, stack.count));
            }
        }
        assert_eq!(
            slots[INVENTORY_SIZE - HOTBAR_SIZE],
            Some(ItemStack::new(Block::GrassBlock, 64))
        );
        assert_eq!(slots[0], None);
    }
}
