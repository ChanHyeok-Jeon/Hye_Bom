from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test():
    return {"message": "Users router 정상 작동"}
