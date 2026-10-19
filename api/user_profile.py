"""User-profile API backed by the PostgreSQL ``user_profiles`` table."""

from datetime import datetime
from enum import Enum
from typing import Annotated

import uvicorn
from fastapi import APIRouter, FastAPI, Form, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from api.config import api_settings
from storage.raw_data.profile import insert_user_profile


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class UserProfileRequest(BaseModel):
    """Profile fields supplied by a user."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_id: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    age: int = Field(ge=0, le=130)
    gender: str = Field(min_length=1, max_length=50)


class UserProfileResponse(BaseModel):
    """Profile returned after it is stored in PostgreSQL."""

    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: str
    email:  EmailStr
    age: int
    gender: Gender
    created_at: datetime


router = APIRouter(prefix="/v1", tags=["user profile"])


@router.post(
    "/user-profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_user_profile(
    user_id: Annotated[str, Form(min_length=1, max_length=200)],
    email: Annotated[EmailStr, Form()],
    age: Annotated[int, Form(ge=0, le=130)],
    gender: Annotated[Gender, Form()],
) -> UserProfileResponse:

    try:
        profile = insert_user_profile(
            user_id=user_id,
            email=str(email),
            age=age,
            gender=gender.value,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return UserProfileResponse.model_validate(profile)

def create_app() -> FastAPI:
    app = FastAPI(title="Fashion Agent API")
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=api_settings.host,
        port=api_settings.port,
    )
