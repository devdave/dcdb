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
