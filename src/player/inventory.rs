//! 玩家物品栏数据模型。
//!
//! 目前物品栏以方块作为物品类型；后续加入工具、食物等物品时，可以把
//! `ItemStack` 的 `Block` 替换成统一的物品注册表 id，而 UI 不需要改变。

use crate::world::block::Block;

pub const HOTBAR_SIZE: usize = 9;
pub const INVENTORY_SIZE: usize = 36;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemStack {
    pub block: Block,
    pub count: u16,
}

impl ItemStack {
    pub const fn new(block: Block, count: u16) -> Self {
        Self { block, count }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inventory {
    pub slots: [Option<ItemStack>; INVENTORY_SIZE],
    pub selected: usize,
    pub open: bool,
}

impl Inventory {
    pub fn new() -> Self {
        let mut slots = [None; INVENTORY_SIZE];
        // 先放一组常用方块，确保进入世界后快捷栏立即可用。
        for (slot, block) in [
            Block::GrassBlock,
            Block::Dirt,
            Block::Stone,
            Block::Cobblestone,
            Block::Sand,
            Block::OakLog,
            Block::OakPlanks,
            Block::Bookshelf,
            Block::RedstoneOre,
        ]
        .into_iter()
        .enumerate()
        {
            slots[INVENTORY_SIZE - HOTBAR_SIZE + slot] = Some(ItemStack::new(block, 64));
        }
        Self {
            slots,
            selected: INVENTORY_SIZE - HOTBAR_SIZE,
            open: false,
        }
    }

    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn select(&mut self, slot: usize) {
        if slot < INVENTORY_SIZE {
            self.selected = slot;
        }
    }

    pub fn selected_stack(&self) -> Option<ItemStack> {
        self.slots[self.selected]
    }

    /// 拿起整个物品堆。物品栏 UI 将返回值作为鼠标光标上的物品。
    pub fn take_stack(&mut self, index: usize) -> Option<ItemStack> {
        self.slots.get_mut(index)?.take()
    }

    /// 拿起一半物品，奇数时让光标多拿一个，和原版左键拆分行为一致。
    pub fn take_half(&mut self, index: usize) -> Option<ItemStack> {
        let slot = self.slots.get_mut(index)?.as_mut()?;
        let count = slot.count.div_ceil(2);
        slot.count -= count;
        let stack = ItemStack::new(slot.block, count);
        if slot.count == 0 {
            self.slots[index] = None;
        }
        Some(stack)
    }

    /// 将光标上的物品放入格子：同种物品尽量堆叠，不同物品则交换。
    pub fn put_stack(&mut self, index: usize, cursor: &mut Option<ItemStack>) {
        let Some(incoming) = cursor.take() else {
            return;
        };
        let Some(slot) = self.slots.get_mut(index) else {
            *cursor = Some(incoming);
            return;
        };

        match slot.as_mut() {
            Some(existing) if existing.block == incoming.block => {
                let available = 64_u16.saturating_sub(existing.count);
                let moved = incoming.count.min(available);
                existing.count += moved;
                if moved < incoming.count {
                    *cursor = Some(ItemStack::new(incoming.block, incoming.count - moved));
                }
            }
            Some(existing) => {
                let old = *existing;
                *existing = incoming;
                *cursor = Some(old);
            }
            None => *slot = Some(incoming),
        }
    }

    /// 右键把光标上的一个物品放入格子。
    pub fn put_one(&mut self, index: usize, cursor: &mut Option<ItemStack>) {
        let Some(incoming) = cursor.as_mut() else {
            return;
        };
        let Some(slot) = self.slots.get_mut(index) else {
            return;
        };
        match slot.as_mut() {
            Some(existing) if existing.block == incoming.block && existing.count < 64 => {
                existing.count += 1;
                incoming.count -= 1;
            }
            None => {
                *slot = Some(ItemStack::new(incoming.block, 1));
                incoming.count -= 1;
            }
            _ => {}
        }
        if cursor.is_some_and(|stack| stack.count == 0) {
            *cursor = None;
        }
    }

    /// 关闭物品栏或拖到面板外时，把光标上的物品尽量放回背包。
    /// 返回无法放回的剩余物品，调用方可继续保留它，避免物品丢失。
    pub fn return_stack(&mut self, mut stack: Option<ItemStack>) -> Option<ItemStack> {
        let incoming = stack.as_mut()?;
        for slot in &mut self.slots {
            let Some(existing) = slot.as_mut() else {
                continue;
            };
            if existing.block != incoming.block || existing.count >= 64 {
                continue;
            }
            let moved = incoming.count.min(64 - existing.count);
            existing.count += moved;
            incoming.count -= moved;
            if incoming.count == 0 {
                return None;
            }
        }
        if let Some(slot) = self.slots.iter_mut().find(|slot| slot.is_none()) {
            *slot = stack.take();
            None
        } else {
            stack
        }
    }

    pub fn take_selected(&mut self) -> Option<Block> {
        let stack = self.slots[self.selected].as_mut()?;
        if stack.count == 0 {
            return None;
        }
        stack.count -= 1;
        let block = stack.block;
        if stack.count == 0 {
            self.slots[self.selected] = None;
        }
        Some(block)
    }

    pub fn give(&mut self, block: Block) {
        if let Some(stack) = self
            .slots
            .iter_mut()
            .filter_map(Option::as_mut)
            .find(|stack| stack.block == block && stack.count < 64)
        {
            stack.count += 1;
            return;
        }
        if let Some(slot) = self.slots.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(ItemStack::new(block, 1));
        }
    }

    /// 为创造模式恢复一组可无限使用的快捷栏方块。
    pub fn refill_hotbar(&mut self) {
        for (slot, block) in [
            Block::GrassBlock,
            Block::Dirt,
            Block::Stone,
            Block::Cobblestone,
            Block::Sand,
            Block::OakLog,
            Block::OakPlanks,
            Block::Bookshelf,
            Block::RedstoneOre,
        ]
        .into_iter()
        .enumerate()
        {
            self.slots[INVENTORY_SIZE - HOTBAR_SIZE + slot] = Some(ItemStack::new(block, 64));
        }
    }
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_a_populated_hotbar() {
        let inventory = Inventory::new();
        assert_eq!(inventory.selected, INVENTORY_SIZE - HOTBAR_SIZE);
        assert_eq!(
            inventory.slots[INVENTORY_SIZE - HOTBAR_SIZE],
            Some(ItemStack::new(Block::GrassBlock, 64))
        );
        assert!(
            inventory.slots[INVENTORY_SIZE - HOTBAR_SIZE..]
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn selecting_outside_inventory_is_ignored() {
        let mut inventory = Inventory::new();
        inventory.select(INVENTORY_SIZE);
        assert_eq!(inventory.selected, INVENTORY_SIZE - HOTBAR_SIZE);
    }

    #[test]
    fn toggle_and_close_work() {
        let mut inventory = Inventory::new();
        inventory.toggle();
        assert!(inventory.open);
        inventory.close();
        assert!(!inventory.open);
    }

    #[test]
    fn selected_stack_can_be_consumed_and_mined_block_returned() {
        let mut inventory = Inventory::new();
        assert_eq!(inventory.take_selected(), Some(Block::GrassBlock));
        assert_eq!(inventory.selected_stack().unwrap().count, 63);
        inventory.give(Block::GrassBlock);
        assert_eq!(inventory.selected_stack().unwrap().count, 64);
    }

    #[test]
    fn stacks_can_be_picked_up_merged_and_swapped() {
        let mut inventory = Inventory::new();
        inventory.slots[0] = Some(ItemStack::new(Block::Stone, 16));
        inventory.slots[1] = Some(ItemStack::new(Block::Stone, 60));
        let mut cursor = inventory.take_stack(0);
        inventory.put_stack(1, &mut cursor);
        assert_eq!(inventory.slots[0], None);
        assert_eq!(inventory.slots[1], Some(ItemStack::new(Block::Stone, 64)));
        assert_eq!(cursor, Some(ItemStack::new(Block::Stone, 12)));

        inventory.slots[2] = Some(ItemStack::new(Block::Dirt, 4));
        inventory.put_stack(2, &mut cursor);
        assert_eq!(inventory.slots[2], Some(ItemStack::new(Block::Stone, 12)));
        assert_eq!(cursor, Some(ItemStack::new(Block::Dirt, 4)));
    }

    #[test]
    fn half_and_one_item_operations_match_vanilla_gestures() {
        let mut inventory = Inventory::new();
        inventory.slots[0] = Some(ItemStack::new(Block::Dirt, 7));
        let mut cursor = inventory.take_half(0);
        assert_eq!(cursor, Some(ItemStack::new(Block::Dirt, 4)));
        assert_eq!(inventory.slots[0], Some(ItemStack::new(Block::Dirt, 3)));
        inventory.put_one(1, &mut cursor);
        assert_eq!(inventory.slots[1], Some(ItemStack::new(Block::Dirt, 1)));
        assert_eq!(cursor, Some(ItemStack::new(Block::Dirt, 3)));
    }

    #[test]
    fn refill_hotbar_restores_creative_mode_blocks() {
        let mut inventory = Inventory::new();
        inventory.slots[INVENTORY_SIZE - HOTBAR_SIZE] = None;
        inventory.refill_hotbar();

        assert_eq!(
            inventory.slots[INVENTORY_SIZE - HOTBAR_SIZE],
            Some(ItemStack::new(Block::GrassBlock, 64))
        );
        assert!(
            inventory.slots[INVENTORY_SIZE - HOTBAR_SIZE..]
                .iter()
                .all(|stack| stack.is_some_and(|stack| stack.count == 64))
        );
    }
}
