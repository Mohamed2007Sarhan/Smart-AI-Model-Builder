# 🤖 Smart AI Model Builder - Automated Model Creation System
## 📸 Screenshots

### 🖼️ Main Interface ( Dark )
![Main Interface](img/Screenshot%202025-10-28%20021301.png)

### 🖼️ Main Interface ( Light )
![Model Selection](img/Screenshot%202025-10-28%20021330.png)

### ⚙️ Feature Setup
![Feature Setup](img/Screenshot%202025-10-28%20021356.png)

### 🚀 Final Output
![Final Output](img/Screenshot%202025-10-28%20021416.png)


## 🌟 Overview

The Smart AI Model Builder is a revolutionary intelligent program that builds, customizes, and packages AI models automatically based on user input. Featuring a modern GUI with multiple themes, this application guides users step-by-step to generate and configure AI models from multiple AI providers including Hugging Face, OpenAI, and Google Gemini.

## 🚀 Key Features

### 1. **Multi-API AI Model Support**
- **Hugging Face Integration**: Access to thousands of pre-trained models
- **OpenAI Support**: Use GPT models and other OpenAI offerings
- **Google Gemini Integration**: Access to Google's powerful AI models
- **Intelligent Model Selection**: Automatically chooses the best model across all providers

### 2. **Smart Multi-Theme GUI Interface**
- **Modern Dark/Light Themes**: Switch between dark and light modes
- **Intuitive Workflow**: Tab-based navigation through Basic Info, Advanced Settings, Progress, and Backups
- **Real-time Logging**: Comprehensive log window showing all operations and progress
- **Visual Feedback**: Progress bar and status indicators for long-running processes

### 3. **Automated Web Information Gathering**
- **Intelligent Search**: Searches the web for relevant information about your model topic
- **Multi-source Data Collection**: Gathers data from multiple trusted sources
- **Smart Analysis**: Analyzes and refines collected data based on your requirements

### 4. **Complete Project Generation**
- **Full Project Structure**: Automatically generates all necessary files in "TINN" folder
- **Training Scripts**: Includes comprehensive training scripts for custom model development
- **Sample Data**: Generates sample datasets to get you started quickly
- **Documentation**: Complete README and configuration files

### 5. **Advanced Dataset Generation**
- **Auto-generation**: Automatically creates training datasets from your descriptions
- **Custom Upload**: Supports uploading your own datasets
- **Multiple Formats**: Works with TXT, CSV, and JSON data formats

### 6. **Model Training Automation**
- **Custom Training**: Train models with your own data
- **Flexible Configuration**: Adjustable training parameters (epochs, batch size, etc.)
- **Model Evaluation**: Built-in evaluation capabilities
- **Model Saving**: Automatically saves trained models for later use

### 7. **Comprehensive Backup System**
- **Automatic Backups**: Creates backups at critical points during model building
- **Restore Functionality**: Easily restore from previous backups
- **Backup Management**: View and manage all backups in the dedicated tab

### 8. **Integrated Chat Interface**
- **Model Testing**: Test your generated models in an interactive chat interface
- **Real-time Responses**: Get immediate feedback from your AI models
- **Conversation History**: Maintain chat history for reference

### 9. **Cross-Platform Installation Scripts**
- **Windows Batch Scripts**: One-click installation and run scripts for Windows
- **Linux/Mac Shell Scripts**: Easy installation and execution on Unix systems
- **Virtual Environment Management**: Automatic virtual environment creation

### 10. **Persistent Configuration Management**
- **JSON Configuration**: All settings saved in app_config.json
- **Automatic Loading**: Previous settings loaded on startup
- **API Key Storage**: Securely stores your API keys for convenience

## 🛠 Technical Implementation

### Language and Frameworks
- **Language**: Python 3.7+
- **GUI Framework**: CustomTkinter (modern, themeable interface)
- **Web Requests**: requests (for API calls and web scraping)
- **Hugging Face Integration**: huggingface-hub (model management)
- **Data Processing**: datasets (for training data handling)
- **Web Scraping**: beautifulsoup4 (information gathering)

### Project Structure
```
Smart AI Model Builder/
├── main.py                 # Main application entry point
├── gui.py                  # Enhanced GUI implementation
├── chat_gui.py             # Chat interface for model testing
├── web_search.py           # Web information gathering
├── model_selector.py       # Multi-API model selection
├── project_generator.py    # Complete project generation
├── requirements.txt        # Python dependencies
├── app_config.json         # Application configuration
├── README.md              # Project documentation
└── test_modules.py        # Comprehensive test suite
```

### Generated Project Structure (TINN/)
```
TINN/
├── src/
│   ├── main.py            # Main application script
│   └── train.py           # Model training script
├── data/                  # Data files
│   ├── input.txt          # Sample input for batch processing
│   ├── output.json        # Sample output from batch processing
│   └── training_data.json # Sample training data
├── models/                # Model files
│   └── trained_model/     # Custom trained models
├── configs/               # Configuration files
├── docs/                  # Documentation
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
├── utils/                 # Utility functions
├── assets/                # Asset files
├── logs/                  # Log files
├── requirements.txt       # Python dependencies
├── install.bat            # Windows installation script
├── install.sh             # Linux/Mac installation script
├── run.bat                # Windows run script
├── run.sh                 # Linux/Mac run script
└── README.md             # Project documentation
```

## 🚀 Installation

### Automatic Installation
1. **Windows**: Run `install.bat`
2. **Linux/Mac**: Run `./install.sh`

### Manual Installation
1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Application
```bash
python main.py
```

### Step-by-Step Process

1. **Basic Information Tab**:
   - Enter your AI Model Name
   - Provide your API Keys (Hugging Face, OpenAI, or Google Gemini)
   - Describe your project requirements in detail

2. **Advanced Settings Tab**:
   - Specify additional APIs
   - Choose dataset generation method (auto or custom upload)
   - Select model size (small, medium, large)
   - Set training time preference (quick, moderate, extensive)

3. **Build Process**:
   - Click "🚀 Build AI Model" to start the automated process
   - Monitor progress in the real-time log
   - Find your generated AI project in the "TINN" folder

4. **Backup Management**:
   - View, create, and restore backups in the Backups tab
   - Automatic backups are created during the build process

5. **Model Testing**:
   - Click "💬 Test Model Chat" to open the chat interface
   - Interact with your generated models in real-time

### Using Generated Projects

#### Interactive Mode
```bash
# Windows
run.bat

# Linux/Mac
./run.sh

# Or manually
python src/main.py
```

#### Batch Processing
```bash
python src/main.py --mode batch --input data/input.txt --output data/output.json
```

#### Training Custom Models
```bash
python src/train.py --data-path data/training_data.json --epochs 5
```

## 📋 Requirements

- Python 3.7 or higher
- At least one API key from:
  - Hugging Face account and API key
  - OpenAI API key
  - Google Gemini API key
- Internet connection for web searches and model downloads

## 📦 Dependencies

All dependencies are listed in [requirements.txt](requirements.txt):
- customtkinter==5.2.0
- requests==2.31.0
- huggingface-hub==0.16.4
- beautifulsoup4==4.12.2
- transformers>=4.10.0
- datasets>=2.0.0
- torch>=1.9.0

## 🧪 Testing

Run the comprehensive test suite to verify all components:
```bash
python test_modules.py
```

## 📖 API Reference

### Main Application (`src/main.py`)
- `--mode`: Run mode (interactive or batch)
- `--input`: Input file for batch mode
- `--output`: Output file for batch mode
- `--model-path`: Path to custom trained model

### Training Script (`src/train.py`)
- `--model-id`: Base model ID (default: auto-selected model)
- `--data-path`: Path to training data (JSON format)
- `--output-dir`: Output directory for trained model
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size

## 🎨 GUI Features

### Basic Info Tab
- Modern input fields with placeholders and default values
- Clear instructions and tips for optimal usage
- Validation for required fields
- Support for multiple API keys

### Advanced Settings Tab
- Dataset options (auto-generation or custom upload)
- Model customization (size and training time preferences)
- Additional API integrations

### Progress Tab
- Real-time logging with timestamped entries
- Visual progress bar with percentage completion
- Status indicators for current operations

### Backups Tab
- List of all available backups
- One-click backup creation and restoration
- Backup management interface

## 🏗 Development Workflow

1. **Information Gathering**: Automatically researches your topic
2. **Dataset Preparation**: Creates or processes training data
3. **Model Selection**: Finds the best model across all available APIs
4. **Customization**: Adapts the model to your requirements
5. **Project Generation**: Creates a complete, runnable project
6. **Backup Creation**: Automatically backs up your work
7. **Validation**: Tests all generated components

## 🔧 Advanced Features

### Multi-API Support
- **Hugging Face**: Access to 100,000+ pre-trained models
- **OpenAI**: GPT models and other OpenAI offerings
- **Google Gemini**: Google's advanced AI models
- **Intelligent Selection**: Chooses the best model across all providers

### Theme Customization
- Switch between dark and light themes
- Consistent appearance across all application windows

### Backup Management
- Automatic backups at critical points
- Manual backup creation
- Easy restoration from any backup

### Chat Interface
- Interactive testing of generated models
- Conversation history preservation
- Real-time AI responses

### Installation Scripts
- Platform-specific installation automation
- Virtual environment management
- Dependency installation

### Configuration Persistence
- All settings saved in JSON format
- Automatic loading of previous settings
- Secure API key storage

## 📅 Generated On
This documentation was automatically generated on 2025-10-28.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for providing access to thousands of pre-trained models
- OpenAI for their powerful GPT models
- Google for Gemini AI models
- CustomTkinter for the modern GUI components
- All contributors to the open-source libraries used in this project

## 🚀 Getting Started Quickly

1. Run `python main.py`
2. Fill in your model requirements and API keys
3. Click "Build AI Model"
4. Your complete AI project will be ready in the TINN folder!
5. Test your model with the integrated chat interface

---

<p align="center">
  <strong>Ready to revolutionize AI model creation? 🚀</strong>

</p>
