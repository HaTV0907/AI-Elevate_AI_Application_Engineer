# add these info to .env
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
# run
source .env
# install requirement packages
pip install flask python-dotenv openai pillow
sudo apt install tesseract-ocr