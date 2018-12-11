# dcdb


Dataclass database library

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


## plans


Swap out @dataclass for my own wrapper so that classes 
are changed in place.


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

record = conn.t.MyWidget.Create(
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
