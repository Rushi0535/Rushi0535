# Simple README Generator with Technical Skills in a Table

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
    "technical_skills": [
        {"category": "Programming Languages Known", "details": "Python 3.6+"},
        {"category": "Python Libraries", "details": "OpenCV, TensorFlow, PyTorch, scikit-image, Detectron2, NLTK, spaCy, Transformers, Gensim, DALL-E, StyleGAN, DeepFaceLab, Keras, Librosa, SpeechRecognition, gTTS, Pydub, Scapy, Requests, Flask"},
        {"category": "AI Frameworks", "details": "TensorFlow, PyTorch, Keras, MXNet, Caffe, Theano"},
        {"category": "AI Algorithms", "details": "Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Transformer Models, BERT, GPT, Object Detection Algorithms (YOLO, SSD, Faster R-CNN), Image Segmentation Algorithms, Face Recognition Algorithms, Pose Estimation Algorithms, Similarity Search Algorithms, Reinforcement Learning Algorithms"},
        {"category": "Operating Systems Used", "details": "Windows, Ubuntu, macOS, CentOS"},
        {"category": "Computer Vision Techniques", "details": "Object Detection, Object Recognition, Image Segmentation, Feature Extraction, Image Enhancement, Image Transformation, Image Filtering, Fourier Transforms, Wavelet Transforms, Image Compression, Color Vision, Pose Estimation, Visual Recognition, Multi-Object Tracking, Facial Recognition, Face Lipsync Processing"},
        {"category": "Web Tools for Development", "details": "Flask, Streamlit, Roboflow, Docker"},
        {"category": "Deep Learning Concepts", "details": "Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) Networks, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Transformers, Attention Mechanisms, Transfer Learning, Reinforcement Learning, Autoencoders, Siamese Networks, Neural Style Transfer"},
        {"category": "IDEs", "details": "Visual Studio Code, PyCharm"},
        {"category": "Python Servers", "details": "WSGI (Web Server Gateway Interface), Nginx"},
        {"category": "Databases", "details": "MySQL, FAISS, Pinecone, ChromaDB, MongoDB"}
    ]
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

readme_content += "\n## 🛠️ Technical Skills\n\n"

# Start of the table
readme_content += "| **Technical Skills**                | **Details** |\n"
readme_content += "|-------------------------------------|-------------|\n"

# Add technical skills to the table
for skill in bio['technical_skills']:
    # Replace any newline characters with spaces
    details = skill['details'].replace('\n', ' ')
    # Handle long text in details by adding line breaks
    details = details.replace(', ', ',<br>')
    readme_content += f"| **{skill['category']}** | {details} |\n"

readme_content += "\n## 🌐 Connect with Me\n"
for platform, link in bio['social'].items():
    readme_content += f"- [{platform}]({link})\n"

readme_content += f"\n## 📫 Let's Chat!\n\nFeel free to reach out to me at **{bio['email']}** for collaborations or discussions!\n"

# Print the README content
print(readme_content)
