from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

# 💡 強化版的 CORS 設定：特別加入了 "OPTIONS"，讓 iPhone 的預檢連線暢通無阻
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"],
)

# 💡 保持延遲載入：先把大腦設為「空」，避免 Render 一開機就因為下載模型而超時
ai_session = None

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...), post_processing: float = Form(0.5)):
    global ai_session 
    
    try:
        # 只有在「第一次有人傳圖片來」的時候，才去載入 u2netp 口袋版大腦
        if ai_session is None:
            print("首次去背：正在載入 AI 模型 (u2netp)...請稍候")
            ai_session = new_session("u2netp")
            
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))

        print("正在執行去背運算...")
        
        output_image = remove(
            input_image,
            session=ai_session,
            alpha_matting=False
        )
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        print("🎉 運算完成！")
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"\n!!! 發生錯誤：{e}")
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    # 讓 Render 動態決定伺服器 Port
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
