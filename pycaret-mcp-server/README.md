# PyCaret MCP Server

🤖 เซิร์ฟเวอร์ MCP สำหรับการทำงานกับ PyCaret อย่างอัตโนมัติ

โปรเจคนี้เป็นส่วนหนึ่งของการบ้าน Week 08 วิชา dstoolbox ที่พัฒนาขึ้นเพื่อให้ AI สามารถใช้งาน PyCaret ผ่าน Model Context Protocol (MCP) ได้อย่างปลอดภัยและมีประสิทธิภาพ

ได้รับแรงบันดาลใจจาก [pandas-mcp-server](https://github.com/marlonluo2018/pandas-mcp-server) แต่ปรับแต่งมาสำหรับการทำงานกับ PyCaret โดยเฉพาะ

## ✨ ความสามารถหลัก

- **เริ่มต้น Experiment:** ตั้งค่า PyCaret สำหรับงาน classification หรือ regression
- **เปรียบเทียบโมเดล:** รันและเปรียบเทียบโมเดล ML หลายตัวพร้อมกัน
- **วิเคราะห์โมเดล:** สร้างกราฟและ metrics เพื่อประเมินประสิทธิภาพ
- **รันโค้ดปลอดภัย:** ป้องกันโค้ดที่เป็นอันตรายด้วยระบบตรวจสอบ

## 🛠️ วิธีติดตั้ง

### สิ่งที่ต้องมี
- Python 3.10 ขึ้นไป
- pip package manager

### ขั้นตอนที่ 1: ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### ขั้นตอนที่ 2: ติดตั้ง MCP Server Runner Extension

1. เปิด VS Code Extensions (Ctrl+Shift+X)
2. ค้นหา "MCP Server Runner"
3. ติดตั้งเป็นอย่างแรก

### ขั้นตอนที่ 3: การตั้งค่า MCP Server Runner

1. กดเปิด MCP Server Runner จาก sidebar
2. กดปุ่ม "+" เพื่อเพิ่ม server ใหม่
3. ตั้งค่าดังนี้:
   - **Name**: pycaret-server
   - **Type**: stdio
   - **Command**: python
   - **Args**: `D:/WK08Homework/pycaret-mcp-server/server.py` (ปรับเส้นทางให้ตรงกับของคุณ)
4. บันทึก configuration

### ขั้นตอนที่ 4: รันด้วย MCP Server Runner

1. ใน MCP Server Runner: ค้นหา "pycaret-server"
2. กดปุ่มเล่น (Play) เพื่อเริ่ม server
3. Server จะเริ่มทำงานและรอการเชื่อมต่อ

### ทางเลือก: ตั้งค่า Claude Desktop

หากต้องการใช้ Claude Desktop แทน:

เพิ่ม configuration นี้ลงในไฟล์ตั้งค่าของ Claude Desktop:

```json
{
  "mcpServers": {
    "pycaret-server": {
      "type": "stdio",
      "command": "python",
      "args": ["D:/WK08Homework/pycaret-mcp-server/server.py"]
    }
  }
}
```

**Configuration File Location:**
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

## 🚀 เครื่องมือที่มีใน MCP Server

### 1. setup_experiment - เริ่มต้นการทดลอง
ใช้สำหรับตั้งค่า PyCaret experiment สำหรับงาน classification หรือ regression

**ตัวอย่างการใช้งาน:**
- โหลดข้อมูลจาก CSV หรือ PyCaret datasets
- กำหนด target variable
- ตั้งค่า session_id สำหรับ reproducibility

### 2. compare_models - เปรียบเทียบโมเดล
รันและเปรียบเทียบโมเดล ML หลายตัวพร้อมกันแบบอัตโนมัติ

**ผลลัพธ์:**
- ตารางเปรียบเทียบ metrics ของทุกโมเดล
- โมเดลที่ดีที่สุดตาม accuracy

### 3. run_pycaret_code - รันโค้ด PyCaret
รันโค้ด PyCaret แบบกำหนดเองอย่างปลอดภัย

**ระบบความปลอดภัย:**
- ตรวจสอบโค้ดก่อนรัน
- บล็อกฟังก์ชันที่เป็นอันตราย
- จำกัดเวลาการทำงาน

### 4. evaluate_model - ประเมินโมเดล
สร้างกราฟและ metrics เพื่อวิเคราะห์ประสิทธิภาพโมเดล

**กราฟที่สร้างได้:**
- AUC curve
- Confusion matrix
- Feature importance

## 📁 Project Structure

```
pycaret-mcp-server/
├── server.py              # MCP server implementation
├── requirements.txt       # Python dependencies
├── core/
│   ├── config.py         # Configuration
│   ├── execution.py      # PyCaret code execution
│   └── evaluation.py     # Model evaluation
├── logs/                 # Application logs
└── README.md            # This file
```

## 🔄 Workflow

1. **Load Data**: Load dataset using pandas or PyCaret datasets
2. **Setup Experiment**: Initialize PyCaret experiment with target variable
3. **Compare Models**: Run compare_models() to find best model
4. **Evaluate**: Generate plots and analyze model performance

## 📊 Example Usage

```python
# Load sample dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# Setup experiment
from pycaret.classification import *
s = setup(data, target='Class variable', session_id=123)

# Compare models
best = compare_models()

# Evaluate
evaluate_model(best)
```

## 📄 License

MIT License - Adapted from pandas-mcp-server

## 🆘 Support

For issues or questions, please refer to the course materials or contact the instructor.
