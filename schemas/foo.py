from pydantic import BaseModel
ROOT_SCHEMA_CLASS = "Foo"

class Foo(BaseModel):
    name: str
