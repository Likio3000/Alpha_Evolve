from pydantic import BaseModel, ConfigDict


class AllowExtrasModel(BaseModel):
    known: int
    model_config = ConfigDict(extra="allow")


def test_allow_extra_fields_are_preserved():
    instance = AllowExtrasModel(known=1, unexpected="value")

    assert instance.known == 1
    assert getattr(instance, "unexpected") == "value"
