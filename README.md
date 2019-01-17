# dcdb


Dataclass database library

## Why another ORM?

I know SQL and have worked with it since the 90's.  I like having the convenience
of having an object transformed from/to a SQL storage engine but I also don't
like having to learn a new pseudo language over and over.

Using dataclass as the container with minimal instrumentation also makes it slightly 
easier to cut down on the amount of work behind the scenes.  As best as possible I want
a wysiwyg interaction.  If I bind a class Foo to the DB, it is stored in the database as
Foo with it's dataclass properties replicated exactly as they were sent to the database.

Lastly, with the transformers relying on capability checking vs typechecking: does a type have a `From` and `To` method vs isinstance(SomeCustomType) there is less to remember.   Just make sure whatever annotation used for a property type is a basic type like 
`str, int, or bool` OR ensure the classs defined a From/To classmethod.

The first and immediate loss is with compatibility.  DCDB is written for sqlite and not another
SQL engine like MySQL, MSSQL, or MySQL.   All of the latter engines may have `SQL` in their
names but they have features and ways of doing the same thing that isn't the same (eg Triggers ).

## status

Heavily under development


## Todo


1. Joins    
2. Transactions  
3. Switch to APSW?
4. Triggers
5. Views?
6. clean this up
7. documentation


## possible plans

1. Swap out @dataclass for my own wrapper so that classes 
are changed in place
2. Concrete classes
3. Anonymouse object generation (eg  SomeTable is not in the registry but allow it to be returned)


## Warranty


There is no guarantee any of this will be compatible with future 
projects as this is still transient



## Usage


```python
import dataclasses as dcs
import dcdb

@dcs.dataclass()
class MyWidget:
    #id: int is automatically added
    name:str
    foo:int
    stuff: dcdb.AutoDict = None
    created_on: int = dcdb.ColumnDef(python=None, database="TIMESTAMP DEFAULT (strftime('%s','now'))")
    

conn = dcdb.DBConnection(":memory:")
"""
  Behind the scenes this creates a table
  
  MyWidget (
    id INTEGER PRIMARY KEY NO DEFAULT, 
    name TEXT NO DEFAULT,
    foo INTEGER NO DEFAULT,
    stuff TEXT DEFAULT NULL,
    created_on TIMESTAMP DEFAULT (strftime('%s','now'))
    )
"""
conn.bind(MyWidget)

record = conn.t.MyWidget(
    name="Bob", 
    foo=123, 
    stuff={"hello":"world", 9:"This is pickled so numeric indexes are integers"}
    )
    
my_record = conn.t.MyWidget.Get("name=?", "Bob") # shortcut for `conn.t.MyWidget.Select("name=?", "Bob").first()`
# my_record is DCDB_MyWidget(MyWidget, dcdb.DBCommonTable)

assert my_record is not None #No effort is made to warn a record wasn't found (eg throwing RecordNotFound or similar)
assert my_record == record #Not necessary, just to show that dataclasses comparison logic is still functioning
assert my_record.stuff == {"hello":"world", 9:"This is pickled so numeric indexes are integers"}

record.delete()
# Note that my_record would still be valid and would undo the prior delete if update() or save() was called.

```

## Copied from sphinx markdown file

# Define a table/model

```
@dataclass
class Foo:
   a_number: int
   text: str

connection = dcdb.DBConnection("file_path/mydb.sqlite3")
connection.bind(Foo)
```

Creates the sqlite table Foo with columns  and .

# Create and retrieve a record

```
@dataclass
class Foo:
   a_number: int
   text: str

connection = dcdb.DBConnection("file_path/mydb.sqlite3")
connection.bind(Foo)

record = connection.tables.Foo(a_number=123, text="Hello world")
# record is automatically saved to database on creation
same_record = connect.tables.Foo.Select("a_number=?", 123)
assert same_record.id == record.id # True
assert same_record.text == "Hello World" # True
```

# DCDB module

## DB DataClass abstraction layer

Version: Super Alpha

turns

```
@dataclass
class Something:
    name: str
    age: int
    species: str = "Human"
```

into

```
CREATE TABLE IF NOT EXISTS Something (
    name TEXT NOT NULL,
    age TEXT NOT NULL,
    species TEXT DEFAULT VALUE Human
)
```

### Quick start

```
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
```

The record has been automatically inserted into the database with a  property set to the relevant row in the
Something table.

### Goals

> 1.  is meant to be a dirt-simple way of saving data to a sqlite table.   Some features, like Joins, are planned
> but that is going to take a while.
> 2.  makes no effort to prevent a user from shooting themselves in the foot.  If you grab two copies of the same
> record, as pointed out above, you can lose data if you lose track of them.
> 3. No dsl’s.   if you have to do  or something crazier like
>  that is insane.  I have known SQL since the 90’s and last thing I want is to learn
> some other quasi-language dialect.
> > * Immediate consequence is a loss of compatibility.
> >   :   MS SQL and MySQL may have SQL in their names but
> >       both have some interesting quirks that make them not friends.

### TODO

1. Cleanup the structure of the package
    * Remove RegisteredTable
    * cut down on __getattr_ calls, if it takes more than one call to reach a resource, that is two much
1. Trim out repetitive parameters

1. Restore positional record/object creation

1. Restore transaction logic

1. Figureout to make AutoList less “goofy”

1. clean up the unit-tests, change naming to test_class/function_case_name

   :   * Split tests apart and make them smaller

       * Review and make classic unit test class’s as appropriate

### Current capabilities

1. Create
1. Select

   :   * alternatively Get can be used to fetch a single record.

1. delete 
1. update/save
1. Relationship helpers through RelationshipFields


```
