import os
import whisper
from flask import Flask, request, render_template_string
from transformers import pipeline
from collections import Counter

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load emotion detection model
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

def analyze_audio(file_path):
    transcription = whisper_model.transcribe(file_path)

    results = []
    previous_emotion = None
    change_points = []

    for segment in transcription["segments"]:
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]

        emotion = emotion_pipeline(text)[0]["label"]

        result = {
            "start_time": format_time(start),
            "end_time": format_time(end),
            "text": text,
            "emotion": emotion
        }

        if emotion != previous_emotion:
            change_points.append({
                "time": format_time(start),
                "emotion": emotion
            })

        previous_emotion = emotion
        results.append(result)

    emotion_counts = Counter([r["emotion"] for r in results])

    return results, change_points, emotion_counts


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    change_points = []
    emotion_counts = {}

    if request.method == "POST":
        audio_file = request.files["audio"]

        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)
        audio_file.save(file_path)

        results, change_points, emotion_counts = analyze_audio(file_path)

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Voice Emotion Analyzer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-dark text-white p-4">

<div class="container">
    <h1 class="text-center">🎤 Voice Emotion Analyzer Dashboard</h1>

    <form method="POST" enctype="multipart/form-data" class="my-4">
        <input type="file" name="audio" class="form-control mb-3" required>
        <button type="submit" class="btn btn-primary">Analyze</button>
    </form>

    {% if results %}

    <h4>Emotion Change Points</h4>
    <ul>
        {% for cp in change_points %}
            <li><b>{{ cp.time }}</b> → {{ cp.emotion }}</li>
        {% endfor %}
    </ul>

    <h4 class="mt-4">Emotion Timeline</h4>
    <canvas id="lineChart"></canvas>

    <h4 class="mt-4">Emotion Distribution</h4>
    <canvas id="pieChart"></canvas>

    <h4 class="mt-4">Detailed Segments</h4>
    <table class="table table-bordered table-light">
        <thead>
            <tr>
                <th>Start</th>
                <th>End</th>
                <th>Emotion</th>
                <th>Text</th>
            </tr>
        </thead>
        <tbody>
            {% for r in results %}
            <tr>
                <td>{{ r.start_time }}</td>
                <td>{{ r.end_time }}</td>
                <td>{{ r.emotion }}</td>
                <td>{{ r.text }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <script>
        const results = {{ results | tojson }};
        const emotionCounts = {{ emotion_counts | tojson }};

        const labels = results.map(r => r.start_time);
        const emotions = results.map(r => r.emotion);

        new Chart(document.getElementById("lineChart"), {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Emotion Timeline",
                    data: emotions,
                    borderWidth: 2,
                    fill: false
                }]
            }
        });

        new Chart(document.getElementById("pieChart"), {
            type: "pie",
            data: {
                labels: Object.keys(emotionCounts),
                datasets: [{
                    data: Object.values(emotionCounts)
                }]
            }
        });
    </script>

    {% endif %}

</div>
</body>
</html>
""", results=results, change_points=change_points, emotion_counts=emotion_counts)


if __name__ == "__main__":
    app.run(debug=True)
