from pydantic import BaseModel, Field

class Foo(BaseModel):
    name: str = Field(..., pattern='^foo$')
