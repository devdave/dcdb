
import dcdb
import dataclasses as dcs

from collections import namedtuple

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

DiffSets = namedtuple("DiffSets", "present,missing")

@dcs.dataclass()
class DiffSets:
    in_database: set
    in_module: set

    @property
    def missing_definitions(self):
        return self.in_database - self.in_module

    @property
    def missing_database(self):
        return self.in_module - self.in_database

    @property
    def no_missing(self):
        return len(self.in_database ^ self.in_module) == 0


class Schema:

    def __init__(self, connection):
        self.connection = connection

    @property
    def tables(self):
        return [row['name'] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]


    def compare(self, scope):

        tables = set(self.tables)
        module_names = set()
        for name in scope.__dict__.keys():
            if name[0] == "_": continue
            if hasattr(getattr(scope, name), "_NO_BIND_") is True: continue
            if hasattr(getattr(scope, name), "__dataclass_fields__") is False: continue
            module_names.add(name)


        return DiffSets(tables, module_names)