"""Unit tests for PostgresConfig URL parsing and configuration."""

from artificial_memory.storage.postgres_store import PostgresConfig


def test_postgres_config_from_url_full():
    url = "postgresql://testuser:secretpass@postgres-host:5433/test_db"
    config = PostgresConfig.from_url(url, min_connections=3, max_connections=15)

    assert config.host == "postgres-host"
    assert config.port == 5433
    assert config.database == "test_db"
    assert config.user == "testuser"
    assert config.password == "secretpass"
    assert config.min_connections == 3
    assert config.max_connections == 15
    assert config.dsn == "postgresql://testuser:secretpass@postgres-host:5433/test_db"


def test_postgres_config_from_url_defaults():
    url = "postgresql://myhost/my_db"
    config = PostgresConfig.from_url(url)

    assert config.host == "myhost"
    assert config.port == 5432
    assert config.database == "my_db"
    assert config.user == "postgres"
    assert config.password == "postgres"
    assert config.dsn == "postgresql://postgres:postgres@myhost:5432/my_db"


def test_postgres_config_from_url_url_encoded_password():
    url = "postgresql://user:p%40ssword@localhost:5432/dbname"
    config = PostgresConfig.from_url(url)

    assert config.user == "user"
    assert config.password == "p@ssword"
