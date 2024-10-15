# Simple README Generator with Technical Skills

bio = {
    "name": "Rushi Prajapati",
    "role": "Code Alchemist",
    "location": "Ahmedabad, India",
    "interests": ["Machine Learning", "Data Science", "AI", "Data Analysis", "Deep Learning", "Gen AI"],
    "tools": ["Python", "TensorFlow", "PyTorch", "React", "Django"],
    "social": {
        "LinkedIn": "https://www.linkedin.com/in/rushi-prajapati12/",
        "Blog Website": "https://rushi-prajapati.medium.com/"
    },
    "projects": [
        {"name": "ANPR", "description": "Description of Project 1", "link": "Link to Project 1"},
        {"name": "Child Abuse Detection", "description": "Description of Project 2", "link": "Link to Project 2"},
        {"name": "Animal Abuse Detection", "description": "Description of Project 3", "link": "Link to Project 3"}
    ],
    "email": "prajapatirushih@gmail.com",
    "technical_skills": {
        "Programming Languages": ["Python 3.6+"],
        "Python Libraries": [
            "OpenCV", "TensorFlow", "PyTorch", "scikit-image", "Detectron2", "NLTK", "spaCy",
            "Transformers", "Gensim", "DALL-E", "StyleGAN", "DeepFaceLab", "Keras", "Librosa",
            "SpeechRecognition", "gTTS", "Pydub", "Scapy", "Requests", "Flask"
        ],
        "AI Frameworks": ["TensorFlow", "PyTorch", "Keras", "MXNet", "Caffe", "Theano"],
        "AI Algorithms": [
            "Convolutional Neural Networks (CNNs)", "Recurrent Neural Networks (RNNs)",
            "Long Short-Term Memory (LSTM)", "Generative Adversarial Networks (GANs)",
            "Variational Autoencoders (VAEs)", "Transformer Models",
            "BERT (Bidirectional Encoder Representations from Transformers)",
            "GPT (Generative Pre-trained Transformer)",
            "Object Detection Algorithms (e.g., YOLO, SSD, Faster R-CNN)",
            "Image Segmentation Algorithms", "Face Recognition Algorithms",
            "Pose Estimation Algorithms", "Similarity Search Algorithms",
            "Reinforcement Learning Algorithms"
        ],
        "Computer Vision Techniques": [
            "Object Detection", "Object Recognition", "Image Segmentation", "Feature Extraction",
            "Image Enhancement", "Image Transformation", "Image Filtering", "Fourier Transforms",
            "Wavelet Transforms", "Image Compression", "Color Vision", "Pose Estimation",
            "Visual Recognition", "Multi-Object Tracking", "Facial Recognition",
            "Face Lipsync Processing"
        ],
        "Web Tools for Development": ["Flask", "Streamlit", "Roboflow", "Docker"],
        "Deep Learning Concepts": [
            "Convolutional Neural Networks (CNNs)", "Recurrent Neural Networks (RNNs)",
            "Long Short-Term Memory (LSTM) Networks", "Generative Adversarial Networks (GANs)",
            "Variational Autoencoders (VAEs)", "Transformers", "Attention Mechanisms",
            "Transfer Learning", "Reinforcement Learning", "Autoencoders", "Siamese Networks",
            "Neural Style Transfer"
        ],
        "Operating Systems Used": ["Windows", "Ubuntu", "macOS", "CentOS"],
        "IDEs": ["Visual Studio Code", "PyCharm"],
        "Python Servers": ["WSGI (Web Server Gateway Interface)", "Nginx"],
        "Databases": ["MySQL", "FAISS", "Pinecone", "ChromaDB", "MongoDB"]
    }
}

# Generate README content
readme_content = f"""# 👋 Hello, World! I'm {bio['name']}, the {bio['role']} 🧪

## 🌟 About Me

I'm a **{bio['role']}** based in **{bio['location']}**.

I'm passionate about {', '.join(['**' + interest + '**' for interest in bio['interests']])}, and I love to tinker with {', '.join(['**' + tool + '**' for tool in bio['tools']])}.

## 💼 My Projects
"""
for project in bio['projects']:
    readme_content += f"- [{project['name']}]({project['link']}): {project['description']}\n"

readme_content += "\n## 🛠️ Technical Skills\n"

for category, skills in bio['technical_skills'].items():
    readme_content += f"\n### {category}\n\n"
    for skill in skills:
        readme_content += f"- {skill}\n"

readme_content += "\n## 🌐 Connect with Me\n"

for platform, link in bio['social'].items():
    readme_content += f"- [{platform}]({link})\n"

readme_content += f"\n## 📫 Let's Chat!\n\nFeel free to reach out to me at **{bio['email']}** for collaborations or discussions!\n"

# Print the README content
print(readme_content)
