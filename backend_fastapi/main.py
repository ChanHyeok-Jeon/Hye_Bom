from fastapi import FastAPI
from routers import analysis, users

app = FastAPI()

# 라우터 등록
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(users.router, prefix="/users", tags=["users"])

@app.get("/")
async def root():
    return {"message": "FastAPI 서버 정상 작동 중입니다."}
