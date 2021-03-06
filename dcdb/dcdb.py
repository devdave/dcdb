# pylint: disable=C0103
"""
    ==============================
    DB DataClass abstraction layer
    ==============================

    Version: Super Alpha

    turns

    .. code-block:: python

        @dataclass
        class Something:
            name: str
            age: int
            species: str = "Human"

    into

    .. code-block:: sql

        CREATE TABLE IF NOT EXISTS Something (
            name TEXT NOT NULL,
            age TEXT NOT NULL,
            species TEXT DEFAULT VALUE Human
        )



    Quick start
    -----------

    .. code-block:: python

        import dcdb
        from dataclasses import dataclass


        @dataclass
        class Something:
            name: str
            age: int
            species: str = "Human"


        connection = dcdb.DBConnection(":memory")
        connection.bind(Something)
        record = connection.tables.Something.Create(name="Bob", age="33", species="Code monkey")

        #To fetch a record, you use pure SQL syntax to make the WHERE clause of a select
        same_record = connection.tables.Something.Get("name=?", "Bob")

        some_record.age = 13

        #Note while record and some_record were the same record
        record.update() # blows away the change to `.age`
        #while
        some_record.update() # would update age to 13



    The record has been automatically inserted into the database with a `.id` property set to the relevant row in the
    Something table.


    Goals
    -----
        1. `dcdb` is meant to be a dirt-simple way of saving data to a sqlite table.   Some features, like Joins, are
         planned but that is going to take a while.

        2. `dcdb` makes no effort to prevent a user from shooting themselves in the foot.  If you grab two copies of the
         same record, as pointed out above, you can lose data if you lose track of them.

        3. No dsl's.   if you have to do `AND(MyColumn==1234, OtherColumn=="abc"))` or something crazier like
        `(NOT(Foo, ILIKE(Bar, "w%"))` that is insane.  I have known SQL since the 90's and last thing I want is to learn
        some other quasi-language dialect.
            * Immediate consequence is a loss of compatibility.
                MS SQL and MySQL may have SQL in their names but
                both have some interesting quirks that make them not friends.


    In progress
    -----

    1. Cleanup the structure of the package
        * Remove RegisteredTable
        * cut down on __getattr_ calls, if it takes more than one call to reach a resource, that is two much
    2. Trim out repetitive parameters
    3. Restore positional record/object creation
    4. Restore transaction logic
    5. Figureout to make AutoList less "goofy"
    6. clean up the unit-tests, change naming to test_class/function_case_name
        * Split tests apart and make them smaller
        * Review and make classic unit test class's as appropriate


    Current capabilities
    --------------------
    1. `.Create`
    2. `.Select`
        * alternatively `.Get` can be used to fetch a single record.
    3. `.delete`
    4. `.update`
    5. SomeTable.[somechild] support through AutoList
    6. other stuff



"""
from __future__ import annotations

import sys
import dataclasses as dcs
import sqlite3
import contextlib
import pickle
import json
import collections
from collections import namedtuple
import enum
import logging
import weakref
import datetime as dt
import inspect
import typing
import abc

LOG = logging.getLogger(__name__)

# Avoid application code from having to import sqlite3
IntegrityError = sqlite3.IntegrityError

@dcs.dataclass
class ColumnDef:
    database: str
    python: object = None

    def From(self, intermediate):
        return intermediate

    def To(self, intermediate):
        return intermediate


@dcs.dataclass
class TableDef:
    """
        Used during the construction/create table phase to
        add index/constraints after column definitions.
    """
    pass



class AbstractTransformedClass(abc.ABC):
    """
    Intended as a integrity checker for other transformer classes.
    """

    @classmethod
    @abc.abstractmethod
    def From(cls, value: str, value_type: typing.Union[object, type]) -> object:
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def To(cls, value: typing.Any, value_type: typing.Union[type, typing.Any]) -> str:
        pass  # pragma: no cover

ConverterPair = namedtuple("ConverterPair", "To,From")

class Transformers:
    """
    Global level registry for transformers

    .. code-block:: python

        import decimal

        Transformer.Set(decimal.Decimal, v,vt: str(v), v,vt: vt(v))

    is one way to ensure that ALL Table Models that have a attribute/column of type `decimal.Decimal`
    are converted to and from the database.

    !NOTE!
    ------

    If decimal.Decimal uses a non-default precision argument, that will be lost in transformation from the database
    and for the moment I don't have an easy solution besides making your own DecimalTransformer class with matching
    To/From classmethods.


    """


    _transforms = {} # type: typing.Dict[str, ConverterPair]

    T = typing.TypeVar("T")
    @classmethod
    def Set(cls, transform_type: typing.Union[type, object]
            , to_func: typing.Callable[[typing.Any, T], typing.AnyStr]
            , from_func: typing.Callable[[typing.Any, T],T[typing.Any]])-> typing.NoReturn:
        cls._transforms[transform_type] = ConverterPair(to_func, from_func)

    @classmethod
    def Has(cls, transform_type: type) -> bool:
        return transform_type in cls._transforms

    # TODO deprecate transformers or rework, not happy having to pass tranform_type everywhere
    @classmethod
    def To(cls, value, transform_type: T) -> str:
        return cls._transforms[transform_type].To(value, transform_type)

    # TODO deprecate transformers or rework, not happy having to pass tranform_type everywhere
    @classmethod
    def From(cls, value: str, transform_type: T) -> T[typing.Any]:
        return cls._transforms[transform_type].From(value, transform_type)


_datetime_format = "%Y-%m-%d %H:%M:%S.%f"
_date_format = "%Y-%m-%d"
Transformers.Set(dt.datetime
                 , lambda v, t: v if isinstance(v, str) else v.strftime(_datetime_format)
                 , lambda v, t: dt.datetime.strptime(v, _datetime_format))
Transformers.Set(dt.date
                 , lambda v, t: v if isinstance(v, str) else v.strftime(_date_format)
                 , lambda v, t: dt.datetime.strptime(v, _date_format).date())
Transformers.Set(dt.time
                 , lambda v, t: v.strftime("%c")
                 , lambda v, t: dt.datetime.strptime(v, "%c").time())


class TransformDatetimeType(AbstractTransformedClass):
    """
    Utility for handling various DateTime formats
    """
    def __init__(self, container, format):
        self.container = container
        self.format = format

    def From(self, value: typing.Any, transform_type: typing.Union[dt.datetime, dt.date, dt.time]):
        if value is None:
            return dt.datetime(0,0,0,0,0,0)

        retval = dt.datetime.strptime(str(value), self.format)

        if transform_type.container == dt.date:
            retval = retval.date()
        elif transform_type.container == dt.time:
            retval = retval.time()

        return retval

    def To(self, value: typing.Any, transform_type: typing.Union[type, object]):
        return value if isinstance(value, str) else value.strftime(self.format)

class DefaultLocalTime(TransformDatetimeType):
    # created_on: dcdb.TransformDatetimeType(datetime.datetime, "%Y-%m-%d T%H:%M") = dcdb.ColumnDef(
    #    "TIMESTAMP DEFAULT (strftime('%Y-%m-%d T%H:%M','now','localtime'))")

    FormatStr = "%Y-%m-%d T%H:%M:%S"
    ColDef = ColumnDef(f"TIMESTAMP DEFAULT (strftime('{FormatStr}','now','localtime'))")

    def __init__(self):
        super().__init__(dt.datetime, self.FormatStr)



def cast_from_database(value: typing.Any, value_type: typing.Union[type, AbstractTransformedClass]):
    """
    Common transformer which ensures that None is None, int is int, etc.

    :param value:
    :param value_type: The type that value must be returned as
    :return: value_type(value)
    """
    LOG.debug(f"cast_from_database(value= {value!r}, value_type= {value_type!r})")
    debug = value

    if value is None:
        retval = None
    elif value_type == str and isinstance(value, str):
        retval = value
    elif value_type in [int, str, float]:
        retval = value_type(value)
    elif value_type == bool:
        retval = bool(int(value))
    elif hasattr(value_type, "From"):
        retval = value_type.From(value, value_type)
    elif Transformers.Has(value_type):
        retval = Transformers.From(value, value_type)
    else:
        ex_msg = f"""
            f"Unable to transform {value_type!r} as returned from DB             
            value = {value!r} 
            debug <= {debug!r} 
            field = {value_type!r}"
        """
        raise ValueError(ex_msg)

    return retval


def cast_to_database(value: typing.Any, value_type: typing.Union[type, AbstractTransformedClass]) -> str:
    """
    Converts the basic types to something compatible with the database

    :param value:
    :param value_type:
    :return str:
    """
    LOG.debug(f"cast_to_datebase-> value: {value!r}, value_type: {value_type!r}")
    debug = value

    if value is None:
        retval = None
    elif value_type == bool:
        retval = int(value)
    elif value_type in [str, int]:
        retval = value_type(value)
    elif value_type == float:
        retval = str(value)
    elif hasattr(value_type, "To"):
        retval = value_type.To(value, value_type)

    elif Transformers.Has(value_type):
        retval = Transformers.To(value, value_type)
    else:
        ex_msg = \
            f"""
                f"Unable to transform {value_type!r} as returned from DB                     
                value = {value!r} 
                debug <= {debug!r} 
                type = {value_type!r}
            """
        raise ValueError(ex_msg)

    return retval


class ColumnTransformer(metaclass=abc.ABCMeta):
    """
    TODO deprecated

    Originally meant as a stub for AutoCast'ed types.
    """
    pass


class FieldPickled(AbstractTransformedClass):
    """

        Given::
        @dataclass
        class Foo:
            flags: FieldPickled = None

        it converts `flags` to a pickled binary string and then back, allowing complex
        dictionary objects to be saved and restored from the database.

        -=======
        Note
        -=======
        Pickle is used vs json because
        `{1:"foo"}` becomes `{"1":"foo"}`

        Why pickle versus JSON?
        1 becomes '1' when going from str to json
        https://bugs.python.org/issue32816

        Supporting Dict->JSON->Dict is trivial and supported by making a JSON AutoCast compatible type.
    """
    SUBTYPE = "BINARY"

    @classmethod
    def From(cls, value, value_type):
        return pickle.loads(value)

    @classmethod
    def To(cls, value: dict, value_type):
        return pickle.dumps(value)


class FieldJSON(AbstractTransformedClass):
    SUBTYPE = "TEXT"

    @classmethod
    def From(cls, value, value_type):
        return json.loads(value)

    @classmethod
    def To(cls, value, value_type):
        return value if isinstance(value, str) else json.dumps(value)


class Fields:
    pickled = FieldPickled
    JSON = FieldJSON


class SQLOperators(enum.Enum):
    AND = "AND"
    OR = "OR"
    NOTEQ = "!="
    LIKE = "LIKE"

@dcs.dataclass()
class TableCriteria:
    name: str
    rowid: str
    cls: DBRegisteredTable = None
    instance: object = None

    @property
    def value(self):
        return self.instance[self.rowid]

    @value.setter
    def value(self, value):
        self.instance[self.rowid] = value


class DescriptorFactory:
    """
        Factory to separate each handler so they don't behave
        as class vars.

    """

    def __init__(self, handler):
        self.handler = handler
        self.args = []
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __get__(self, instance, owner):

        if instance is None:
            return self

        handler = self.handler(*self.args, **self.kwargs)
        handler.set_owner(instance)
        return handler



class ListSelect(collections.abc.Sequence):

    def __init__(self, child_name, relationship_field=None, parent_join_field="id", where=None, add_set=None,
                 remove_set=None):

        if relationship_field is None:
            if "." not in child_name:
                raise ValueError(
                    f"Provided {child_name} but missing {relationship_field}.  Is a '.' missing from {child_name}?")
            else:
                child_name, relationship_field = child_name.split(".")

        self.tables = None
        self.__where = where
        self._add_set = add_set
        self._remove_set = remove_set

        self.child_name = child_name
        self.relationship_field = relationship_field
        self.child = None  # type: DBCommonTable

        self.parent_name = None
        self.parent_join_field = parent_join_field
        self.parent = None

    def set_owner(self, owner):
        self.parent = owner
        self.tables = owner.tables

    @property
    def _where(self):


        return f"{self.relationship_field}=?" \
            if self.__where is None \
            else f"{self.relationship_field}=? AND {self.__where}"

    def sum(self, sum_field, where = None, *args, default=None):
        where_sql = None
        args = [self.parent[self.parent_join_field], *args]

        where_sql = None
        if where is not None:
            if self._where is not None:
                where_sql = " AND ".join([self._where, where])
            else:
                where_sql = " AND ".join([where])
        elif self._where is not None:
            where_sql = " AND ".join([self._where])

        record = None
        column = f"SUM({sum_field})"
        if where_sql:
            record = self.child_cls.Select(where_sql, *args, columns=column)
        else:
            record = self.child_cls.Select(columns=column)

        result = record.fetchone(direct_record=True)
        if result is None:
            return default
        else:
            return result[0]


    def where(self, clause, *args, offset=None, limit=None):

        try:
            original_where = self.__where
            self.__where = clause

            return self.child_cls.Select(self._where
                                         , self.parent[self.parent_join_field]
                                         , *args
                                         , limit_offset=offset
                                         , limit_count=limit)
        finally:
            self.__where = original_where


    def __get__(self, instance, owner):
        if instance is None:
            return self


        self.parent = instance
        self.tables = instance.tables

        return self

    @property
    def child_cls(self):
        return self.tables[self.child_name]

    def _select(self, offset, limit=None) -> DBCursorProxy:

        return self.child_cls.Select(self._where, self.parent[self.parent_join_field], limit_offset=offset,
                                     limit_count=limit)

    def __getitem__(self, index: int):

        return self._select(index, limit=1).fetchone()

    def __iter__(self)->typing.Generator[DBCommonTable]:
        for record in self._select(0, limit=-1):
            yield record

    def __setitem__(self, _: int, record: DBCommonTable):
        raise TypeError("Assigning records to a specific index is not supported with ListSelect")

    def __iadd__(self, other):
        # TODO really need Transactions
        for record in other:
            record[self.relationship_field] = self.parent[self.parent_join_field]
            if self._add_set:
                record[self._add_set[0]] = self._add_set[1]
            record.save()

        return self

    def __delitem__(self, key: int):
        record = self._select(key, limit=1).fetchone()
        record.delete()

    def pop(self, index):

        if isinstance(index, int):
            record = self._select(index).fetchone()
        else:
            record = index

        record[self.relationship_field] = None
        if self._remove_set:
            record[self._remove_set[0]] = self._remove_set[1]

        record.save()
        return record

    def remove(self, record):
        record[self.relationship_field] = None
        if self._remove_set:
            record[self._remove_set[0]] = self._remove_set[1]

        record.save()
        return record

    def __len__(self):
        return self.child_cls.Count(self._where, self.parent[self.parent_join_field])

    def insert(self, *records: DBCommonTable):
        for record in records:
            record[self.relationship_field] = self.parent[self.parent_join_field]
            record.save()

    def create(self, **kwargs):
        kwargs[self.relationship_field] = self.parent[self.parent_join_field]
        return self.child_cls(**kwargs)

    def first(self) -> DBCommonTable:
        """
        Shortcut helper to fetch the first related record

        TODO deprecate?
        :return:
        """
        return self._select(0).fetchone()


class DictSelect(collections.abc.MutableMapping):

    def __init__(self, child_name, name_field=None, relationship_field=None, parent_join_field="id", by_order=None):

        if relationship_field is None:
            if "." not in child_name:
                raise TypeError(f"child_field is None but child_table {child_name} "
                                "field does not match table.field format")
            else:
                relationship_field = name_field
                child_name, name_field = child_name.split(".")

        self.tables = None

        self.child_name = child_name
        self.name_field = name_field
        self.relationship_field = relationship_field
        self.child_cls = None
        self.by_order = by_order

        self.parent_name = None
        self.parent_join_field = parent_join_field
        self.parent = None

    def set_owner(self, owner):
        self.parent = owner
        self.tables = owner.tables
        self.child = self.tables[self.child_name]

    def __get__(self, instance: DBCommonTable, owner: type):
        if instance is None:
            return self

        if self.parent is None:
            self.parent = instance
            self.tables = instance.tables
            self.child = self.tables[self.child_name]

        return self

    def __contains__(self, item):
        record = self.child.Select(f"{self.relationship_field}={self.parent[self.parent_join_field]}"
                                 f" AND {self.name_field}=?", item
                                 , order_by=self.by_order).fetchone()

        return record is not None

    def __getitem__(self, key):
        return self.child.Select(f"{self.relationship_field}={self.parent[self.parent_join_field]}"
                                 f" AND {self.name_field}=?", key
                                 , order_by=self.by_order).fetchone()

    def __setitem__(self, key, record: DBCommonTable):
        record[self.relationship_field] = self.parent[self.parent_join_field]
        record.save()

    def add(self, record):
        record[self.relationship_field] = self.parent[self.parent_join_field]
        record.save()

    def create(self, **new_record_args):
        new_record_args[self.relationship_field] = self.parent[self.parent_join_field]
        return self.child(**new_record_args)

    def __delitem__(self, key):
        record = self[key]
        record[self.relationship_field] = None
        record.save()

    def __iter__(self):
        for record in self.child.Select(f"{self.relationship_field}={self.parent[self.parent_join_field]}"):
            yield record

    def __len__(self):
        return self.child.Count(f"{self.relationship_field}={self.parent[self.parent_join_field]}")


class LeftNamedMultiJoin(collections.abc.MutableMapping):
    """
        Relationship handler for many to many relationships.
    """

    """
    Select * FROM {self.child.table} 
    LEFT JOIN {self.join_table} jt ON jt.{self.join_right_rowid} = {self.child.table}.{self.child.key}
    WHERE
        jt.{left_rowid} = '{parent[parent_rowid]}'
                    
    """

    def __init__(self, join_table, join_left_rowid, join_right_rowid, child_table, child_name_field, parent_rowid="id", child_rowid="id"):

        #"Box2Thing", "box_id", "thing_id", "Thing", child_table_key="name"

        self.join_table = TableCriteria(join_table, None)
        self.join_left_rowid = join_left_rowid
        self.join_right_rowid = join_right_rowid

        self.child = TableCriteria(child_table, child_rowid)
        self.child_name_field = child_name_field

        self.parent = TableCriteria(None, parent_rowid)
        self.parent_id = None

        """
        Select * FROM {self.child.table} 
        LEFT JOIN {self.join_table} jt ON jt.{self.join_right_rowid} = {self.child.table}.{self.child.key}
        WHERE
            jt.{left_rowid} = '{parent[parent_rowid]}'
                        
        """


    def set_owner(self, owner):
        self.parent.instance = owner
        self.parent_id = self.parent.value
        self.join_table.cls = owner.tables[self.join_table.name]
        self.child.cls = owner.tables[self.child.name]

    def _joined_select(self, discriminator=None, columns=None):

        args = []
        where_clause = f"jt.{self.join_left_rowid} = {self.parent_id}"
        if discriminator:
            args.append(discriminator)
            where_clause += f" AND {self.child.name}.{self.child_name_field} = ?"

        if columns is None:
            columns = f"{self.child.name}.*"

        sql = f"""
        SELECT {columns} FROM {self.child.name}
        LEFT JOIN {self.join_table.name} jt ON jt.{self.join_right_rowid} = {self.child.name}.{self.child.rowid}
        WHERE
        {where_clause}             
        """.strip()
        try:
            return self.parent.instance._meta_.connection.execute(sql, args)
        except sqlite3.OperationalError:
            LOG.exception(f"SQL {sql!r}")
            raise

    def __len__(self):
        return self._joined_select(columns="COUNT(*)").fetchone()[0]


    def __contains__(self, item: typing.Union[str, DBCommonTable])->bool:
        """
        Checks if item is a str or record instance

        checks the relevant table to see if it contains the provided item

        :param item:
        :return bool:
        """

        if isinstance(item, DBCommonTable):
            key = item[self.child_name_field]
        else:
            key = item

        return self._joined_select(discriminator=key).fetchone() is not None


    def keys(self):
        return [r[0] for r in self._joined_select(columns=self.child_name_field).fetchall()]


    def __getitem__(self, key: typing.Union[str, DBCommonTable]):

        if isinstance(key, DBCommonTable):
            key = key[self.child_name_field]

        raw_record = self._joined_select(discriminator=key).fetchone()
        record = self.child.cls.Assign(**raw_record)
        return record


    def __delitem__(self, key):
        record = self[key]
        SQL = f"DELETE FROM {self.join_table.name} WHERE {self.join_left_rowid} = ? AND {self.join_right_rowid} = ?"
        result = self.parent.cls.connection.execute(SQL, (self.parent_id, record[self.child.key]))


    def __iter__(self):
        for result in self._joined_select():
            yield self.child.cls.Assign(**result)

    def __setitem__(self, key: str, child_record: DBCommonTable):
        kwargs = {
            self.join_left_rowid: self.parent_id
            , self.join_right_rowid: child_record[self.child.rowid]
        }

        assert key == child_record[self.child_name_field]
        join_record = self.join_table.cls(**kwargs)

    def insert(self, *records):
        for record in records:
            self[record[self.child_name_field]] = record


    def __iadd__(self, other: [DBCommonTable]):
        for record in other:
            self[record[self.child_name_field]] = record

        return self

    def add(self, new_child):
        self[new_child[self.child_name_field]] = new_child


class LocalOne2One:

    def __init__(self, owner_rowid: str, target_name_fk: str):
        #("target_id", "TargetTable.id")

        self.__owner = TableCriteria(None, owner_rowid)
        self.__target = TableCriteria(*target_name_fk.split("."))


    def set_owner(self, record):
        self.__owner.instance = record
        self.__target.cls = record.tables[self.__target.name]


    def __getattr__(self, item):
        where_str = f"{self.__target.rowid}={self.__owner.value}"
        record = self.__target.cls.Get(where_str)
        return getattr(record, item)

    def __get__(self, owner: DBCommonTable, objtype: DBCommonTable = None):


        self.__owner.instance = owner
        self.__target.cls = owner.tables[self.__target.name]

        return self

        if self.__owner.value is not None:
            where_str = f"{self.__target.rowid}={self.__owner.value}"
            return self.__target.cls.Get(where_str)
        else:
            return None

    def __set__(self, owner, target_record):

        assert isinstance(target_record, DBCommonTable) is True
        self.__owner.value = target_record[self.__target.rowid]
        owner.save()
        return self

    def set(self, target_record):
        assert isinstance(target_record, DBCommonTable) is True
        self.__owner.value = target_record[self.__target.rowid]
        self.__owner.instance.save()
        return self

    def clear(self):
        self.__owner.value = None
        self.__owner.instance.save()

    def delete(self):
        self.clear()
        where_str = f"{self.__target.rowid}={self.__owner.value}"
        target_record = self.__target.cls.Get(where_str)  # type: DBCommonTable
        target_record.delete()



    def __delete__(self, owner_table):
        where_str = f"{self.__target.rowid}={self.__owner.value}"
        target_record = self.__target.cls.Get(where_str) # type: DBCommonTable
        target_record.delete()

        self.__owner.value = None
        owner_table.save()


        return self


class _RelationshipFields:
    """
        Consolidates Relationship handlers into one place.

        Each attribute is a property descriptor so that
        each :class:`DescriptorFactory` is a unique instance per class definition.

    """

    @property
    def dict(self)->DictSelect:
        return DescriptorFactory(DictSelect)

    @property
    def unordered_list(self)->ListSelect:
        return DescriptorFactory(ListSelect)

    @property
    def local_one_to_one(self)->LocalOne2One:
        return DescriptorFactory(LocalOne2One)

    @property
    def named_left_join(self)->LeftNamedMultiJoin:
        return DescriptorFactory(LeftNamedMultiJoin)

RelationshipFields = _RelationshipFields()



class DBConnection:
    """
        Database connection manager

        Responsible for keeping a reference/handle of a sqlite3 connection
        and a table registry (which maps to said database)
    """

    __slots__ = ("closed",
                 "dburl",
                 "schema",
                 "_conn_",
                 "_tables", "tables", "t", "registry",
                 "_dirty_records", "_dirty_records_track_changes",
                 # I find this so bizarre in a way but it makes a sense, of sorts, that I need to add this
                 "__weakref__"
                 )

    _sql_list_tables = """SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"""

    _sql_describe_table = """
        PRAGMA table_info(%s)
    """

    def __init__(self, dburl: str = None, schema_name = "main"):

        self.closed = False
        self.dburl = dburl
        self.schema = schema_name

        if dburl is not None:
            self._conn_ = sqlite3.connect(str(dburl))
            self._conn_.row_factory = sqlite3.Row
            self._conn_.isolation_level = None

        registry = TablesRegistry(self)
        self.registry = registry
        self._tables = registry
        setattr(self, "tables", weakref.proxy(self._tables))
        setattr(self, "t", weakref.proxy(self._tables))
        self.t = registry
        self._dirty_records = set()
        self._dirty_records_track_changes = False

    def purge(self, this_deletes_everything=False):
        if this_deletes_everything is not True:
            raise RuntimeError("Purge called without confirmation `this_deletes_everything` set to True")

        for record in self._conn_.execute(self._sql_list_tables):
            self._conn_.execute(f"DROP TABLE {record['name']}")

        self._conn_.execute("VACUUM")

    def close(self):
        if self.closed is not True:
            del self.t
            del self._tables
            del self.registry
            del self._dirty_records
            self._conn_.close()
            self._conn_ = None
            del self._conn_

        self.closed = True

    def _dirty_record_track_changes(self, flag: bool):
        self._dirty_records_track_changes = flag
        if flag is False:
            self._dirty_records = set()

    @property
    def dirty_records_tracking(self):
        return self._dirty_records_track_changes

    def _dirty_record_add(self, record: DBCommonTable):
        if self._dirty_records_track_changes:
            self._dirty_records.add(record)
            return True
        else:
            return False

    def _dirty_record_remove(self, record: DBCommonTable):
        if record in self._dirty_records:
            self._dirty_records.remove(record)

    @contextlib.contextmanager
    def track_changes(self):
        try:
            self._dirty_record_track_changes(True)
            yield
        finally:
            for table in {x for x in self._dirty_records}:
                table.update()
                table._dirty_reset()
            self._dirty_record_track_changes(False)

    def binds(self, *tables: [DBCommonTable], create_table: bool = True):
        return [self.bind(table, create_table=create_table) for table in tables]

    def bind(self, *tables, create_table: bool = True, schema = "main"):

        schema = schema if schema is not None else self.schema
        collect = []
        for table in tables:
            # TODO, this makes binding fragile, what if dataclasses internals change?
            # reference  https://github.com/python/cpython/blob/3.7/Lib/dataclasses.py#L187
            if hasattr(table, "__dataclass_fields__") is False:
                raise TypeError(f"Expected {table} to be a dataclass, missing __dataclass_fields__ attribute")

            bound_class = self.registry.mk_bound_dataclass(table, table.__name__, schema=schema)

            if create_table:
                bound_class._DRV.Create_table(self, bound_class)

            collect.append(bound_class)

        if len(tables) == 1:
            return collect[0]
        else:
            return collect

    def bind_scan(self, scope, ignore: list = None, create_table=True) -> None:
        """
            Convenience helper to bind every dataclass in the provided scope object to
            the current connection.

            ####
            NOTE!
            ####

            It is important to note that "every" is literally every dataclass, each class will have SQL tables created on
            the connection.   This can pollute a database with tables it doesn't need and are never
            used.

            add the property `__NO_BIND__ = False` to skip over non-modal classes

        :param scope: A dictionary or module object
        :param ignore: A list (or better a set) of class names to ignore when binding the contents of scope
        :param create_table: Should tables be created (if not exists) for each bound dataclass
        :return:  None
        """
        ignore = [] if ignore is None else ignore

        elements = scope.__dict__.items() if hasattr(scope, "__dict__") else scope.items()
        for name, val in elements:
            if name in ignore: continue
            if name.startswith("_"): continue
            if inspect.isclass(val):
                if hasattr(val, "_NO_BIND_"): continue
                if hasattr(val, "__dataclass_fields__"):
                    self.bind(val, create_table=create_table)

    def handle(self):
        return self._conn_

    def direct(self, sql, *args):
        return self._conn_.cursor().execute(sql, *args) if args else self._conn_.cursor().execute(sql)

    # @property
    # def tables(self):
    #     return self._tables

    @property
    def cursor(self):
        return self._conn_.cursor()

    @property
    def execute(self):
        return self._conn_.cursor().execute


class DBRegisteredTable:
    # TODO kill this off and or consolidate somehow
    __slots__ = ('connection', 'bound_cls', 'table_name', 'fields', "__weakref__")

    def __init__(self, connection, bound_cls, table_name, fields):
        self.connection = connection
        self.bound_cls = bound_cls
        self.table_name = table_name
        self.fields = fields

    def __getattr__(self, item):
        return getattr(self.bound_cls, item)

    def __call__(self, *args, **kwargs)->DBCommonTable:
        # TODO, profile this to see how the unit tests are using it.
        if "id" not in kwargs:
            return self.bound_cls.Create(*args, **kwargs)
        else:
            raise ValueError("Do not assign record ID through DBRegisteredTable")
            return self.bound_cls(*args, **kwargs)

    def Assign(self, **kwargs) ->DBCommonTable:
        return self.bound_cls(**kwargs)

    def Insert_many(self, *column_sets):
        collection = []
        for column_fields in column_sets:
            try:
                collection += [self.bound_cls.Create(**column_fields)]
            except sqlite3.IntegrityError as ex:
                raise ValueError(f"IntegrityError on {column_fields} - {ex}")
            except Exception as ex:
                raise ValueError(f"{type(ex)} excception on {column_fields} - {ex}")

        return collection

    def Count(self, where=None, *args, **kwargs):
        kwargs['params'] = "COUNT(*)"
        cursor = self._DRV.Select(self.connection, self.bound_cls, where, *args, columns="COUNT(*)")
        return cursor.fetchone()[0]

    def Update(self, where, **values):
        return DBSQLOperations.Update(self.connection, self.bound_cls, self.fields, where, **values)


class DBSQLOperations:
    """
        Placeholder for a DB driver for other sqlite API's or even for other SQL services.
    """

    @classmethod
    def Create_table(cls, connection, table_cls, safeguard_if_not_exists: bool = True):
        """
            table_definitions - None column definitions like foreign keys and constraints

        """
        body = ""
        column_map = dict()
        type_map = {int: "INTEGER", str: "TEXT", bool: "BOOLEAN"}

        table_def_match = lambda field_val: field_val.type is TableDef and isinstance(field_val.default, str)
        table_definitions = {f.name: f.default for f in dcs.fields(table_cls) if table_def_match(f)}

        for field in dcs.fields(table_cls):
            if field.name == "id": continue  # this is ALWAYS part of the def so don't bother
            if field.type is None: continue
            if field.type is TableDef: continue
            column_map[field.name] = cls._Create_column(field, type_map)

        body_elements = ["id INTEGER PRIMARY KEY NOT NULL"]
        body_elements += [f"{k} {v}" for k, v in column_map.items()]

        # sql.append(", ".join(), )

        if table_definitions:
            body_elements += [str(td) for td in table_definitions.values()]

        sql = f"""CREATE TABLE {"IF NOT EXISTS" if safeguard_if_not_exists else ""} 
                {table_cls._meta_.schema_table} 
                ({", ".join(body_elements)})
        """.strip()

        try:
            return connection.execute(sql)
        except sqlite3.OperationalError as ex:
            LOG.exception(ex)
            LOG.error("Failed query:\n", f"{sql!r}:{len(sql)}")
            raise

    @classmethod
    def _Create_column(cls, column_field: dcs.Field, type_map: {type: str}) -> str:
        # TODO make a test just for this method

        default_value = dcs.MISSING
        sql_column = None

        def find_default(column_field):
            return column_field.default_factory() \
                if column_field.default_factory is not dcs.MISSING \
                else column_field.default

        if isinstance(column_field.default, ColumnDef):
            return column_field.default.database

        # TODO find a better way to handle this
        if column_field.type not in type_map:
            if hasattr(column_field.type, "SUBTYPE"):
                sql_column = column_field.type.SUBTYPE


            elif inspect.isclass(column_field.type) and issubclass(column_field.type, enum.Enum):
                if issubclass(column_field.type, enum.IntEnum):
                    sql_column = "INT"
                else:
                    sql_column = "TEXT"

            else:
                sql_column = "TEXT"
        else:
            sql_column = type_map[column_field.type] if column_field.type in type_map else "TEXT"

        default_value = find_default(column_field)

        if default_value is not dcs.MISSING:
            if default_value is None:
                default_value = "Null"
            else:
                default_value = cast_to_database(default_value, column_field.type)

            sql_column += f" DEFAULT {default_value}"
        else:
            sql_column += " NOT NULL"

        # TODO

        return sql_column

    @classmethod
    def Insert(cls, cursor, tbl_cls, **column_fields):

        dc_fields = tbl_cls._meta_.fields
        if column_fields:
            keys = list(column_fields.keys())
            sql = [f"INSERT INTO {tbl_cls._meta_.schema_table}",
                   "(", ", ".join(keys), ")"
                                         "VALUES",
                   "(", ", ".join(["?" for x in range(0, len(keys))]), ")"
                   ]
            sql = " ".join(sql).strip()

            try:
                values = [cast_to_database(cf_value, dc_fields[cf_name].type) for cf_name, cf_value in
                          column_fields.items()]
            except KeyError as ex:
                raise RuntimeError(f"Missing column/{ex} - perhaps a typo? Choices are {dc_fields.keys()}")

            return cursor.execute(sql, values)

        else:
            sql = f""" INSERT INTO {tbl_cls._meta_.schema_table} DEFAULT VALUES"""
            return cursor.execute(sql)

    @classmethod
    def Select(cls, cursor, tbl_cls, where=None, *where_vals, **params):

        append = []

        # Might seem odd but this catches two conditions in one
        # 1. is the key in params and 2. is params[key] Truthy versus False or None

        if params.get("order_by", False):
            #TODO need some sort of mechanism to decide ASC or DESC order
            append += [f"ORDER BY {params['order_by']} DESC"]

        if params.get("limit_offset", False):
            if params.get("limit_count", False):
                append += [f"LIMIT {params['limit_count']} OFFSET {params['limit_offset']}"]
            else:
                append += [f"LIMIT -1 OFFSET {params['limit_offset']}"]
        elif params.get("limit_count", False):
            append += [f"LIMIT {params['limit_count']}"]


        append = " ".join(append)

        sql = f"""
            SELECT
                {params['columns'] if "columns" in params else '*'}
            FROM {tbl_cls._meta_.schema_table}
            {"WHERE" if where else ""}
                {where if where else ''}
            {append}
        """.strip()

        if where and where_vals:
            assert where_vals is not None, f"Where has ? but no arguments for {where}!"
            return cursor.execute(sql, where_vals)
        else:
            return cursor.execute(sql)

    @classmethod
    def Update(cls, connection, tbl_cls, dc_fields, where, **values: dict):

        field_values = {}
        params = {k[1:]: v for k, v in values.items() if k[0] == "_"}

        for k, v in {k: v for k, v in values.items() if k.startswith("_") is False}.items():
            if k not in dc_fields: continue
            field_values[k] = cast_to_database(v, dc_fields[k].type, )

        if isinstance(where, list):
            where_sql = where[0]
            where_values = where[1:]
        else:
            where_values = []
            where_sql = where

        sql = f"""
            UPDATE
                {tbl_cls._meta_.schema_table}
            SET
                {", ".join([f"{k}=?" for k in field_values.keys()])}
            WHERE
                {where_sql}
            {f"LIMIT {params['limit']}" if "limit" in params else ""}
            {params['append'] if 'append' in params else ""}
        """

        # print(sql, list(field_values.values()) + list(where_values))
        return connection.execute(sql, list(field_values.values()) + list(where_values))

    @classmethod
    def Delete(cls, connection, tbl_cls, where: str, where_vals: list = None, limit: int = 1):

        assert where, f" Where {where!r} is empty, use '1=1' for a total table purge"
        sql = f"""
            DELETE FROM
                {tbl_cls._meta_.schema_table}
            WHERE {where}
        """
        # TODO apparently my sqlite build does NOT have order by/limit support with DELETE
        # this slightly disturbs me
        # https://www.sqlite.org/compile.html#enable_update_delete_limit

        # print(repr(sql), repr(where), repr(where_vals), repr(limit))
        if where_vals:
            return connection.execute(sql, **where_vals)
        else:
            return connection.execute(sql)

    @classmethod
    def CreateIndex(cls, connection, tbl_cls, index_fields, index_name=None, is_unique=True):
        SQL = ['CREATE']
        if is_unique is True:
            SQL.append('UNIQUE')

        SQL.append('INDEX')

        if index_name is None:
            index_name = "idx_" + "_".join([field[:4] for field in index_fields])

        SQL.append(f"ON {tbl_cls._meta_.schema_table}")
        SQL.append(f"({','.join(index_fields)})")
        SQL = " ".join(SQL)
        return connection.execute(SQL)


class DBDirtyRecordMixin:

    def _init_dirty_record_tracking_(self):
        self._dirty_record = False
        self._dirty_record_ignore = True
        self._locked = True

    def __hash__(self):
        """
        Hashes records by their garbage collection reference ID.

        Only obvious edge case is if I have two separate instances of the same record and
        they both get caught up in the dirty records list.   If I have record A and B of the same database record
        but B is updated after A, changes to A will be lost.   Otherwise I am hard pressed to see any other immediate
        problems.


        :return: hash()
        """

        return hash(id(self))

    @property
    def _is_dirty(self):
        return self._dirty_record

    def _set_dirty(self, key=None, value=None):

        if key in self._meta_.fields:
            if self._meta_.connection._dirty_record_add(self):
                object.__setattr__(self, "_dirty_record", True)

    def _dirty_reset(self):
        self._meta_.connection._dirty_record_remove(self)
        object.__setattr__(self, "_dirty_record", False)

    def __setattr__(self, key, value):
        if key in self._meta_.fields:
            self._set_dirty(key, value)

        if getattr(self, "_locked", False) is True and key not in self._meta_.fields:
            raise ValueError(f"Record is locked, trying to set {key} when not in fields: {self._meta_.fields.keys()}")

        super().__setattr__(key, value)
        return value


class DBMeta:
    __slots__ = ("connection", "name", "fields", "indexes", "schema")

    def __init__(self, connection, name, fields, indexes, schema=None):
        self.connection = connection
        self.name = name
        self.fields = fields
        self.indexes = indexes
        self.schema = schema

    @property
    def schema_table(self):
        return f"{self.schema}.{self.name}"


class AttrDict:

    def __init__(self):
        super().__setattr__("contents", {})

    def __setattr__(self, key, value):
        self.contents[key] = value

    def __setitem__(self, key, value):
        self.contents[key] = value

@dcs.dataclass
class DBCommonTable(DBDirtyRecordMixin):
    """
    :cvar _conn_: DBConnection
    :cvar _table__: str
    :cvar _original_: object
    :cvar _fields_:[dcs.Field]
    """

    id: int  # TODO should I make this an initvar?

    _meta_ = None # type: DBMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # super().__post_init__()



    def __post_init__(self):

        LOG.debug("Finalization with __post_init__")

        # Post processing to convert from DB to Application
        for field in self._meta_.fields.values():
            if field.name == "id": continue

            # avoid mutation tracking as much as possible
            orig_value = getattr(self, field.name)
            value = cast_from_database(orig_value, field.type)
            super().__setattr__(field.name, value)


        self._init_dirty_record_tracking_()


    def __setitem__(self, key, value):
        if key in self._meta_.fields:
            setattr(self, key, value)

    def __getitem__(self, key):
        if key in self._meta_.fields:
            return getattr(self, key)
        else:
            raise ValueError(f"No {key} in {self}")

    def __str__(self):
        return self._meta_.name

    @classmethod
    def DB(cls):
        return cls._meta_.connection.cursor

    @classmethod
    def Select(cls, where=None, *args, **kwargs):
        cursor = cls.DB()
        cls._DRV.Select(cursor, cls, where, *args, **kwargs)
        return DBCursorProxy(cls, cursor)

    @classmethod
    def Get(cls, where, *where_vals, **kwargs):
        return cls.Select(where, *where_vals, **kwargs).fetchone()

    @classmethod
    def Create(cls, **kwargs):
        cursor = cls.DB()
        cls._DRV.Insert(cursor, cls, **kwargs)
        cls._DRV.Select(cursor, cls, f"id={cursor.lastrowid}")
        return DBCursorProxy(cls, cursor).fetchone()

    def delete(self):
        cursor = self._DRV.Delete(self._meta_.connection, self, f"id={self.id}")

        for field in dcs.fields(self):
            setattr(self, field.name, None)
        return cursor

    def save(self):
        values = dcs.asdict(self)
        rec_id = values.pop("id", None)

        retval = None

        if rec_id is not None:
            retval = self._DRV.Update(self._meta_.connection, self, self._meta_.fields, f"id={rec_id}",
                                      **values)
        else:
            retval = self._DRV.Insert(self._meta_.connection, self, self._meta_.fields, **values)
            self.id = retval.lastrowid

        self._dirty_reset()

    def update(self):
        values = dcs.asdict(self)
        rec_id = values.pop("id")
        cursor = self._DRV.Update(self._meta_.connection, self, self._meta_.fields, f"id={rec_id}",
                                  **values)
        self._dirty_reset()
        return cursor.rowcount == 1


class DBCursorProxy:
    """
        Wrap around the cursor instance so that record
        requests are instantiated into the appropriate

    """

    def __init__(self, factory: DBCommonTable, cursor: sqlite3.Cursor):
        self._factory = factory
        self._cursor = cursor

    def fetchone(self, direct_record=False) -> DBCommonTable:
        row = self._cursor.fetchone()
        if row is None: return None

        if direct_record is True:
            return row
        else:
            return self._factory(**row) if row is not None else None

    def fetchmany(self, size=100) -> DBCommonTable:
        # todo figure out why , size=sqlite3.Cursor.arraysize is a member descriptor (sign of a bad slots somewhere)
        rows = self._cursor.fetchmany(size)
        while rows:
            for row in rows:
                yield self._factory(**row)
            else:
                rows = self._cursor.fetchmany(size)
                if not rows:
                    return

    def fetchall(self) -> DBCommonTable:
        for row in self._cursor.fetchall():
            yield self._factory(**row) if row is not None else None

    def __getattr__(self, key):
        return getattr(self._cursor, key)

    def __iter__(self)-> typing.Generator[DBCommonTable]:
        for row in self.fetchmany():
            yield row


class TablesRegistry:
    """

        Responsible for converting dataclasses into DBDataclasses and keeping
        a registry of these new classes.

        TODO, creates a new dataclass and then creates a slotted version.   Perhaps
        a reconnect mechanism would be a good idea to save the class definitions for
        environments like flask where bind maybe called more than once.


    """

    def __init__(self, connection):
        self._registry: {str: DBCommonTable} = dict()
        self._connection = connection

    def __del__(self):
        for name in [name for name in self._registry.keys()]:
            del self._registry[name]

        self._registry = {}

    def mk_bound_dataclass(self, source_cls, name: str, schema = None) -> DBCommonTable:
        """

        :param source_cls: a dataclass intended to be used for working with a databasee
        :param name:  Generally is cls.__name__
        :return: <type cls name=name, MRO=(cls,DBCommonTable>
        """

        def eval_type(st):
            return st if not isinstance(st, str) else eval(st, vars(sys.modules[source_cls.__module__]))

        # TODO put this in a higher scope or use a common heritage sentinel class
        exclusion_list = [TableDef]

        db_cls = dcs.make_dataclass(f"DCDB_{name}", [], bases=(source_cls, DBCommonTable))
        db_cls_fields = dcs.fields(db_cls)

        for field in db_cls_fields:
            try:
                field.type = eval_type(field.type)
            except TypeError as ex:
                raise TypeError(f"Bad column type {field.type!r} for {field.name!r} on {name!r} - {ex}")
            except NameError as ex:
                raise NameError(f"Bad column type {field.type!r} for {field.name!r} on {name!r} class table")

        db_cls_fields = [f for f in db_cls_fields if f.type not in exclusion_list]

        # TODO - Figure out why __hash__ from DCCommonTable is being stripped out
        setattr(db_cls, "__hash__", lambda instance: hash(id(instance)))
        db_cls = self.ioc_assignments(db_cls, source_cls, name, db_cls_fields, schema=schema)

        self._registry[name] = DBRegisteredTable(self._connection, db_cls, name, db_cls._meta_.fields)
        setattr(self, name, self._registry[name])  # test to cut down on __getattr__ calls
        return db_cls

    def ioc_assignments(self, db_cls: DBCommonTable, source_cls, name, db_cls_fields, schema=None) -> DBCommonTable:

        assert schema is not None, f"Which database schema does {name} belong to?"
        def set_default(name, value):
            if getattr(db_cls, name, None) is None:
                setattr(db_cls, name, value)

        fields = {f.name: f for f in db_cls_fields}
        set_default("_meta_", DBMeta(self._connection, name, fields, {}, schema=schema))
        set_default("tables", self)
        set_default("t", self)
        set_default("_original_", source_cls)
        set_default("_DRV", DBSQLOperations)

        return db_cls

    def get_table(self, table_name: str) -> DBCommonTable:
        if (table_name in self._registry) is False:
            raise RuntimeError(f"Missing table {table_name!r} requested.  Tables available: {self._registry.keys()!r}")
        return self._registry.get(table_name)

    __getattr__ = get_table
    __getitem__ = get_table
    # def __getattr__(self, key):
    #     return self.get_table(key)
    #
    # def __getitem__(self, key):
    #     return self.get_table(key)
