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
            Block::OakLeaves,
            Block::Glass,
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
}
