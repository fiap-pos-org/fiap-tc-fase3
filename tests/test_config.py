from hospital_assistant import config


def test_root_dir_aponta_para_a_raiz_do_repo():
    assert (config.ROOT_DIR / "pyproject.toml").is_file()


def test_diretorios_de_dados_ficam_dentro_de_data():
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.EVAL_DIR):
        assert path.parent == config.DATA_DIR
