#pylint: disable=C0111,C0112,C0301,C0103,W0621,C0102
#TODO remake .pylintrc?  I only want some of these disables for tests though
"""
    Tests for DCDB
"""

import pytest

import dcdb

from dataclasses import dataclass, fields, field
import enum

#debug


@pytest.fixture
def connection():
    """
        Fixture used to speed up testing
    """
    return dcdb.DBConnection(":memory:")

@pytest.fixture()
def conn2(connection:dcdb.DBConnection):
    """

    :param connection: dcdb.DBConnection uses the connection fixture
    :return:
    """

    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)

    return connection

@dataclass
class Widget:
    # __table__ = "widgets"
    name: str
    age: int
    panacea: bool = False #in retrospect this makes no sense but I am just going to keep using this word


def test_dcdb_cast_to_database_type__fix_ordering_error():
    """
    Odd test but written to isolate a bug

    """

    value = "Bob"
    result = dcdb.cast_to_database(value, str)
    assert isinstance(result, str)
    assert value == result


def test_dbconnection___connects():
    my_conn = dcdb.DBConnection(":memory:")
    result = my_conn.handle().execute("SELECT 1=1").fetchone()
    assert result[0] == 1



def test_dcdb_cast_to_database_AND_autocastdict___works():

    import pickle

    @dataclass
    class Foo:
        colors: dcdb.AutoCastDict

    test_value = {"red": 1, "blue": 2, "complicated":[1, 2, "this isn't a color"]}
    test_subject = Foo(test_value)

    foo_fields = fields(Foo)
    colors_field = fields(Foo)[0]
    result = dcdb.cast_to_database(test_value, colors_field.type)
    assert isinstance(result, bytes)
    assert result == pickle.dumps(test_value)


def test_dcdb_cast_to_database():

    class Switch(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value):
            return value.name

        @classmethod
        def From(cls, value):
            return cls.__members__[value]


    test = Switch.OFF
    expected = "OFF"
    actual = dcdb.cast_to_database(test, Switch)
    assert expected == actual


def test_tableregistry_mk_bound(connection):

    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)
    res = connection.handle().execute("SELECT name FROM sqlite_master WHERE type='table' and name = 'Widget' ORDER BY name;")
    name = res.fetchone()
    assert name[0] == "Widget"
    assert connection.registry.Widget is not None
    rows = connection.execute("PRAGMA table_info(Widget)").fetchall()

    assert rows[0]['name'] == "id"
    assert rows[0]['type'] == "INTEGER"

    assert rows[1]['name'] == "name"
    assert rows[1]['type'] == "TEXT"

    assert rows[2]['name'] == "age"
    assert rows[2]['type'] == "INTEGER"

    #Sqlite coerces bools to intege
    # which is kind of a stretch as basically sqlite coerces everything into a string

    assert rows[3]['name'] == "panacea"
    assert rows[2]['type'] == "INTEGER"

def test_dbcommontable_ishashable(conn2):

    x = conn2.t.Widget(name="Bob", age=10, panacea=True)
    y = conn2.t.Widget(name="Alive", age=12, panacea=False)
    x2 = conn2.t.Widget.Get("name=?", "Bob")
    assert (y == x) is False
    assert (x == x) is True
    assert (x == x2) is True
    records = set()
    assert x.__hash__ is not None
    records.add(x)
    records.add(y)

def test_dbconnection_binds_multiple_tables(connection):

    @dataclass()
    class Foo:
        name: str

    @dataclass()
    class Bar:
        name: str
        num: int

    @dataclass()
    class Bob:
        last_name: str
        age: int

    (MyFoo, MyBar, MyBob,) = connection.binds(Foo, Bar, Bob)

    for (test_cls, db_cls,) in ((Foo, MyFoo,), (Bar, MyBar,), (Bob, MyBob,)):
        assert issubclass(db_cls, test_cls)
        assert issubclass(db_cls, dcdb.DBCommonTable)


def test_make_complex_widget(connection):

    from typing import ClassVar
    @dataclass
    class ComplexWidget:

        #by definition of how dataclasses work, these cannot be empty/unassigned/null so don't bother telling sqlite to enforce that.
        #
        name: str
        age: int

        NameAgeUnique:dcdb.TableDef = "CONSTRAINT NameAgeUnique UNIQUE(name,age)"

    connection.bind(ComplexWidget)
    res = connection.handle().execute("""PRAGMA index_info(sqlite_autoindex_ComplexWidget_1)""")

    index_rows = res.fetchall()
    """
        [0, 1, 'name']
        [1, 2, 'age']
    """
    assert len(index_rows) == 2
    assert index_rows[0][2] == "name"
    assert index_rows[1][2] == "age"


def test_db_dataclass_throws_error_on_missing_param(conn2):

    with pytest.raises(TypeError):
        conn2.t.Widget(id=1)

    with pytest.raises(TypeError):
        conn2.t.Widget(id=1, name="Bob")

    with pytest.raises(TypeError):
        conn2.t.Widget(id=1, name="Bob", age=55)

    record = conn2.t.Widget(id=1, name="Bob", age=55, panacea=True)
    assert record.id == 1
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True


# def test_DBTableRegistry__create(conn2):
#
#
#     with pytest.raises(dcdb.IntegrityError):
#         conn2.t.Widget(name="Bob", age=55)
#
#     conn2.t.Widget(name="Bob", age=55, panacea=False)
#     row = conn2.execute("SELECT * FROM Widget LIMIT 1").fetchone()
#
#     assert row['name'] == "Bob"
#     assert row['age'] == 55


def test_DBTableProxy_InsertMany(conn2):

    MyWidget = conn2.t.Widget
    MyWidget.Insert_many(
        dict(name="Bob", age=10, panacea=True),
        dict(name="Joe", age=22, panacea=False),
        dict(name="Steve", age=35, panacea=True)
    )

    record = conn2.execute("SELECT COUNT(*) FROM Widget").fetchone()
    assert record[0] == 3

    bob = MyWidget.Get("name=?", "Bob")
    Joe = MyWidget.Get("name=?", "Joe")
    Steve = MyWidget.Get("name=?", "Steve")
    assert bob.age == 10
    assert Joe.name == "Joe" #this seems redundent

def test_DBCommonTable_AND_DBRegistry__create_with_default_factory(connection):

    @dataclass()
    class Test:
        foo: str = "Hello"
        bar: int = dcdb.ColumnDef(python=None, database="TIMESTAMP DEFAULT (strftime('%s','now'))")

    connection.bind(Test)


    instance = connection.t.Test.Create()


def test_DBTableRegistry__create(conn2):

    with pytest.raises(TypeError):
        record = conn2.t.Widget(id=1, name="Bob", age=55)

    record = conn2.t.Widget(name="Bob", age=55, panacea=True)
    record.save()
    assert record.id == 1
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True

    record = conn2.t.Widget(name="Bob", age=55, panacea=False)
    record.save()
    assert record.id == 2
    assert record.panacea == False



def test_DBConnection__data_integrity_from_sqlite(conn2):
    with pytest.raises(dcdb.IntegrityError):
        conn2.execute("INSERT INTO Widget(name,age) VALUES ('Bob', 33)")


def test_DBCommonTable__Select(conn2):

    conn2.execute("INSERT INTO Widget(name,age,panacea) VALUES ('Bob', 33, 1)")
    row = conn2.t.Widget.Select("name=?", "Bob").fetchone()

    assert row.name == "Bob"
    assert row.age == 33
    assert row.panacea == 1


def test_Insert_Select_Get(conn2):

    MyWidget = conn2.t.Widget
    MyWidgetClass = MyWidget.bound_cls

    MyWidget(name="Bob", age=44, panacea=False)

    record = MyWidget.Select("name=?", "Bob").fetchone()

    assert record.name  == "Bob"
    assert record.age == 44
    assert record.panacea == False

    record = MyWidget.Get("name=?", "Bob")
    isinstance(record, MyWidgetClass)

    assert record.name == "Bob"
    assert record.age == 44
    assert record.panacea is False


def test_DBCommonTable_update(conn2):
    conn2.bind(Widget)

    # TODO should I consider it a problem that setting id skips saving the actual record?
    #  record = conn2.t.Widget(id=10, name="Joe", age=22, panacea=True)
    record = conn2.t.Widget(name="Joe", age=22, panacea=True)
    assert record.name == "Joe"
    assert record.age == 22
    assert record.panacea is True

    record.panacea = False
    record.age = 33
    record.update()
    same_record = conn2.t.Widget.Get("id=?", record.id)
    assert isinstance(same_record, conn2.t.Widget.bound_cls)
    assert same_record.panacea is False
    assert same_record.age == 33
    assert same_record.id == record.id


def test_DBCommonTable_Update__bulk_update(conn2):

    # Populate the db
    conn2.t.Widget(name="Bob", age=13, panacea=True)
    conn2.t.Widget(name="Joe", age=22, panacea=True)
    conn2.t.Widget(name="Sue", age=34, panacea=True)
    conn2.t.Widget(name="Mary", age=44, panacea=True)
    conn2.t.Widget(name="It", age=9999, panacea=True)

    # Bulk update roughly half
    cursor = conn2.t.Widget.Update(["age<?", 35], panacea=False)

    #Sanity check they are correct
    assert conn2.t.Widget.Get("name=?", "Bob").panacea is False
    assert conn2.t.Widget.Get("name=?", "Joe").panacea is False
    assert conn2.t.Widget.Get("name=?", "Sue").panacea is False
    assert conn2.t.Widget.Get("name=?", "Mary").panacea is True
    assert conn2.t.Widget.Get("name=?", "It").panacea is True




def test_DBTable_handles_dicttype(connection):

    @dataclass
    class Foo:
        bar: dcdb.AutoCastDict
        name: str

    @dataclass()
    class Bar:
        name: str
        blank: dcdb.AutoCastDict = None


    connection.bind(Foo)
    connection.bind(Bar)

    test_value = {"a":1, 2:"b", "three":"c"}
    # expected =   {"a":1, 2:"b", "three":"c"}

    record = connection.t.Foo(bar=test_value, name="Test thing")

    assert record.bar == test_value, repr(record.bar)

    record2 = connection.t.Bar(name="Bob")

    assert record2.blank == None

    record3 = connection.t.Bar(name="Bob", blank=test_value)

    assert record3.blank == test_value


def test_DBCommonTable_hashing_equals(conn2):

    record1 = conn2.t.Widget(name="Alice", age=35, panacea=False)
    record2 = conn2.t.Widget.Get("name=?", "Alice")
    record3 = conn2.t.Widget.Get("age=?", 35)
    assert record1.id == record2.id
    assert record1 == record2
    assert record2 == record3
    assert record3 == record1

    @dataclass()
    class Foo:
        height:int
        width:int

    conn2.bind(Foo)
    record4 = conn2.t.Foo(height=12, width=22)
    record5 = conn2.t.Foo.Get("height=?",12)
    record6 = conn2.t.Foo.Get("width=?", 22)
    assert record4 != record1
    assert record5 != record2
    assert record6 != record3
    assert record6 == record4
    assert record4 == record5
    assert record5 != record3



def test_table_registry(connection):
    MyWidget = connection.bind(Widget)
    SameWidget = connection.t.Widget
    assert issubclass(MyWidget, Widget)
    assert issubclass(SameWidget.bound_cls, Widget)

def test_TablesRegistry_missing_table(connection):

    with pytest.raises(RuntimeError):
        connection.t.NotATable


def test_table_direct_children(connection):

    @dataclass()
    class Parent:
        name:str

        @property
        def children(self):
            return self.tables.Child.Select(f"parent_id={self.id}")


        def create_new_child(self):
            child = self.tables.Child(parent_id=self.id)
            child.update()
            return child

        def disown(self, child):
            child.delete()

    @dataclass()
    class Child:

        parent_id:int
        def __str__(self):
            return "Child of {self.parent_id} #{self.id}"

        @property
        def parent(self):
            return self.tables.Parent.Get("id=?", self.parent_id)


    MyParent = connection.bind(Parent)
    MyChild = connection.bind(Child)

    bob = connection.t.Parent(name="Bob")
    bob.create_new_child()
    bob.create_new_child()
    bob.create_new_child()

    children = connection.execute("SELECT * FROM Child").fetchall()
    assert len(children) == 3

    redheaded_stepchild = connection.t.Child.Get("id=2")
    assert redheaded_stepchild.parent.name == "Bob"
    bob.disown(redheaded_stepchild)


    children = connection.execute("SELECT * FROM Child").fetchall()
    assert len(children) == 2



def test_autoselect_property(connection):

    @dataclass()
    class Foo:
        name: str
        some_num: int

    @dataclass()
    class Bar:
        name: str
        other_num: int
        parent_id: int = None

        parent = dcdb.AutoSelect("Foo", "id", "parent_id")

    connection.binds(Foo, Bar)

    parent_record = connection.t.Foo(name="Bert", some_num=1234)
    child_record = connection.t.Bar(name="Brad", other_num=56)

    child_record.parent = parent_record
    child_record.update()

    assert child_record.parent_id == parent_record.id
    assert child_record.parent.name == parent_record.name

    del child_record.parent
    assert child_record.parent_id is None



def test_error_columns_mismatch(conn2):

    @dataclass
    class OtherWidget:
        length: int
        height: int
        weight: int
        name: str

    conn2.bind(OtherWidget)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)
    assert thingamajig.name == "Thinga 2000"
    assert thingamajig.length == 10
    assert thingamajig.height == 1
    assert thingamajig.weight == 8


def test_mutation_tracking(conn2:dcdb.DBConnection):


    @dataclass
    class OtherWidget:
        length:int
        height:int
        weight:int
        name:str

    conn2.bind(OtherWidget)
    bob = conn2.t.Widget(name="Bob", age=44, panacea=False)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)

    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False

    #Known and intended gotcha
    thingamajig.height = 5
    assert thingamajig._is_dirty is False

    with conn2.track_changes():

        bob.age = 22
        assert bob._is_dirty is True
        assert len(conn2._dirty_records) == 1

        thingamajig.weight = 12
        thingamajig.name = "New Hotness 2.0"
        assert len(conn2._dirty_records) == 2
        assert thingamajig._is_dirty is True

    assert len(conn2._dirty_records) == 0
    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False


    bob_copy = conn2.t.Widget.Get("id=?", bob.id)
    thing_copy = conn2.t.OtherWidget.Get("id=?", thingamajig.id)

    assert bob_copy.age == 22
    assert thingamajig.weight == 12
    assert thingamajig.name == "New Hotness 2.0"
    # Intentional and intended, change tracking is only available when explicitly tracking
    assert thingamajig.height == 5


def test_column_as_enum(connection):

    import enum

    class EnumFlags(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value):
            return value.name

        @classmethod
        def From(cls, value):
            return cls.__members__[value]

    @dataclass()
    class Test:

        enum_column:EnumFlags = EnumFlags.OFF


    connection.bind(Test)
    default_record = connection.t.Test()
    off_record = connection.t.Test(enum_column=EnumFlags.OFF)
    on_record = connection.t.Test(enum_column=EnumFlags.ON)





def test_AutoList_full(connection):
    import enum

    class ChildTableStatus(enum.IntEnum):
        PENDING = 0
        COMPLETE = 1

        @classmethod
        def To(cls, value):
            return value.value

        @classmethod
        def From(cls, value):
            return cls(int(value))

    @dataclass()
    class Parent:

        name: str

        Steves = dcdb.AutoList(["Parent", "id"], ["Children", "parent_id"])
        Steves.where(
            "name LIKE 'Steve%'",
        )

        Pending = dcdb.AutoList(["Parent", "id"], ["Children", "parent_id"])
        Pending.where(
            "parent_id = {parent.id}",
            ["status = {0}", ChildTableStatus.PENDING.value]
        )

        @Pending.creator
        def Pending(self, parent_record, child_table, **kwargs):
            kwargs['status'] = ChildTableStatus.PENDING.value
            kwargs['parent_id'] = parent_record.id
            return child_table.Insert(**kwargs)

        @Pending.adder
        def Pending(self, parent_record, child_record):
            child_record['parent_id'] = parent_record.id
            child_record.update()
            return child_record

        @Pending.remover
        def Pending(self, parent_table, child_record):
            print(self, parent_table, child_record)
            child_record['parent_id'] = None
            child_record.update()
            return child_record

        Complete = dcdb.AutoList(["Parent", "id"], ["Children","parent_id"])
        Complete.where(
            "parent_id = {parent.id}",
            ["status = {0}", ChildTableStatus.COMPLETE.value]
        )


    @dataclass()
    class Children:
        name:str
        parent_id:int = None
        status:ChildTableStatus = ChildTableStatus.PENDING

    connection.binds(Parent, Children)

    steve, stjr, joe, joejr, \
        carl, alice, smith = \
        connection.t.Children.Insert_many(
        {"name":"Steve"},
        {"name": "Steve JR."},
        {"name": "Joe"},
        {"name": "Joe Jr."},
        {"name": "Carl", "status":ChildTableStatus.COMPLETE},
        {"name": "Alice", "status":ChildTableStatus.COMPLETE},
        {"name": "Smith", "status":ChildTableStatus.COMPLETE},
    )


    bob = connection.t.Parent(name="Bob")

    completed_count = connection.t.Children.Count("status=?", ChildTableStatus.COMPLETE.value)
    pending_count = connection.t.Children.Count("status=?", ChildTableStatus.PENDING.value)
    assert completed_count == 3
    assert pending_count == 4

    result = bob.Pending
    assert len(result) == 0
    bob.Pending.add(steve, stjr, joe, joejr)
    assert len(bob.Pending) == 4
    bob.Pending.remove(joe)
    assert len(bob.Pending) == 3
    assert len(bob.Steves) == 2

    bob.Complete.add(carl, alice, smith)
    assert len(bob.Complete) == 3
    bob.Complete.remove(alice)
    assert len(bob.Complete) == 2


def test_AutoList_dotted_relations(connection):

    @dataclass()
    class Boss:
        name: str
        employees = dcdb.AutoList("Boss.id", "Employee.boss_id")

    @dataclass()
    class Employee:
        name:str
        age:int
        boss_id:int = None

    connection.bind_scan(locals())

    boss = connection.t.Boss.Create(name="Bill")

    emp1 = connection.t.Employee.Create(name="Joe",age=12)
    emp2 = connection.t.Employee.Create(name="Al", age=22)
    emp3 = connection.t.Employee.Create(name="Jane", age=30)
    emp4 = connection.t.Employee.Create(name="Friday",age=28)

    boss.employees.add(emp1, emp2, emp4)



def main():
    import sys
    import pathlib as pl



    my_path = pl.Path(__file__).resolve().parent
    sys.path.append(str(my_path))


    pytest.main()

if __name__ == "__main__":
     main()