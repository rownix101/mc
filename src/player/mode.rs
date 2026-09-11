//! 玩家游戏模式。

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GameMode {
    #[default]
    Survival,
    Creative,
}

impl GameMode {
    pub const fn toggle(self) -> Self {
        match self {
            Self::Survival => Self::Creative,
            Self::Creative => Self::Survival,
        }
    }

    pub const fn display(self) -> &'static str {
        match self {
            Self::Survival => "生存模式",
            Self::Creative => "创造模式",
        }
    }

    pub const fn is_creative(self) -> bool {
        matches!(self, Self::Creative)
    }

    /// 存档中的模式编号: 0 = 生存, 1 = 创造.
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Survival => 0,
            Self::Creative => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GameMode;

    #[test]
    fn toggles_between_survival_and_creative() {
        assert_eq!(GameMode::Survival.toggle(), GameMode::Creative);
        assert_eq!(GameMode::Creative.toggle(), GameMode::Survival);
    }

    #[test]
    fn save_ids_are_stable() {
        assert_eq!(GameMode::Survival.as_u8(), 0);
        assert_eq!(GameMode::Creative.as_u8(), 1);
    }
}
