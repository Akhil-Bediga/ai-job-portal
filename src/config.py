import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MySQLConfig:
    host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    port: int = int(os.getenv("MYSQL_PORT", "3306"))
    user: str = os.getenv("MYSQL_USER", "jobs_user")
    password: str = os.getenv("MYSQL_PASSWORD", "jobs_pass")
    database: str = os.getenv("MYSQL_DATABASE", "jobportal")

    @property
    def url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )


@dataclass(frozen=True)
class MongoConfig:
    uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    database: str = os.getenv("MONGO_DB", "jobportal")


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data").lstrip("./")
    api_port: int = int(os.getenv("API_PORT", "5002"))


mysql_cfg = MySQLConfig()
mongo_cfg = MongoConfig()
pipeline_cfg = PipelineConfig()
