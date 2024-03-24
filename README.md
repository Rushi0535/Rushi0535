# 👋 Hello, World! I'm Rushi Prajapati, the Code Alchemist 🧪

```python
class Bio:
    def __init__(self):
        self.name = "Rushi Prajapati"
        self.role = "Code Alchemist"
        self.location = "Ahmedabad,India"
        self.interests = ["Machine Learning", "Data Science", "AI", "DATA Analysis", "Deep Learning", "Gen AI"]
        self.tools = ["Python", "TensorFlow", "PyTorch", "React", "Django"]
        self.social = {
            "LinkedIn": "https://www.linkedin.com/in/rushi-prajapati12/",
            "Blog Website": "https://rushi-prajapati.medium.com/"
        }

    def showcase_projects(self):
        projects = [
            {"name": "ANPR", "description": "Description of Project 1", "link": "Link to Project 1"},
            {"name": "Child Abuse Detection", "description": "Description of Project 2", "link": "Link to Project 2"},
            {"name": "Animal Abuse Detection", "description": "Description of Project 3", "link": "Link to Project 3"}
        ]
        return projects

    def get_in_touch(self):
        email = "prajapatirushih@gmail.com"
        return f"Feel free to reach out to me at {email} for collaborations or discussions!"

# Instantiate Bio
bio = Bio()

# Print Bio
print(f"👋 Hi there! I'm {bio.name}, the {bio.role} based in {bio.location}.")
print("\n### 🌟 About Me")
print(f"I'm passionate about {', '.join(bio.interests)}, and I love to tinker with {', '.join(bio.tools)}.")
print("\n### 💼 My Projects")
for project in bio.showcase_projects():
    print(f"- [{project['name']}]({project['link']}): {project['description']}")
print("\n### 🌐 Connect with Me")
for platform, link in bio.social.items():
    print(f"- [{platform}]({link})")
print("\n### 📫 Let's Chat!")
print(bio.get_in_touch())
