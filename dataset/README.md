# Dataset - Manual Collection

## 📁 Folder Structure

```
dataset/
├── images/
│   ├── ai/                    # AI-generated website screenshots
│   │   ├── framer_001.png
│   │   ├── wix_001.png
│   │   └── ...
│   └── human/                 # Human-coded website screenshots
│       ├── human_001.png
│       ├── human_002.png
│       └── ...
└── labels.csv                 # Dataset labels
```

## 📋 Manual Collection Process

### 1. Screenshot Naming Convention
- **AI websites**: `{tool}_{number}.png` (e.g., `framer_001.png`, `wix_002.png`)
- **Human websites**: `human_{number}.png` (e.g., `human_001.png`, `human_002.png`)

### 2. Screenshot Requirements
- **Resolution**: 1920x1080 or higher
- **Format**: PNG or JPG
- **Content**: Full-page screenshots (not just viewport)
- **Quality**: Clean, professional appearance, avoid popups/overlays

### 3. Labels.csv Format
```csv
id,tool
framer_001,framer
wix_001,wix
human_001,human
human_002,human
```

### 4. Collection Steps
1. Find AI-generated websites (Framer, Wix ADI, Notion, etc.)
2. Take screenshots using browser dev tools
3. Save with proper naming in correct folder (`ai/` or `human/`)
4. Add entry to `labels.csv`

## 🎯 Target Dataset
- **50+ AI-generated websites** (Framer, Wix, Notion, etc.)
- **50+ Human-coded websites** (Professional, personal portfolios)
- **Total**: 100+ samples

## 📊 Current Status
Check `labels.csv` for current dataset statistics. 