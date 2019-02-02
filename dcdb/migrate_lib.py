
import dcdb
import dataclasses as dcs

@dcs.dataclass()
class Migration:
    current_stage: str
    ran_at: dcdb.TransformDatetimeType


class Migrator:
    # God module anti-pattern

    def __init__(self, cwd, connection):
        self.cwd = cwd
        self.connection = connection

    def checkdb(self):
        pass


class Schema:

    def __init__(self, connection):
        self.connection = connection

    @property
    def tables(self):
        return [name for name in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]

