from app.utils.database import get_database_file_path, initialize_database


def test_initialize_database_creates_path() -> None:
    initialize_database()
    db_path = get_database_file_path()
    assert db_path.exists()
