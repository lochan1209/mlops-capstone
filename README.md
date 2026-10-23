# 1. Build the Docker image
docker build -t student-performance:v1 .

# 2. Run the container detached, mapping port 8077 on host to 8000 in container
docker run -d --name sp-test -p 8077:8000 student-performance:v1

# 3. Check health
curl -s http://localhost:8077/health

# 4. Run sample prediction
curl -s -X POST http://localhost:8077/predict -H "Content-Type: application/json" \
    -d '{"gender":"female","race_ethnicity":"group B","parental_level_of_education":"bachelor'\''s degree","lunch":"standard","test_preparation_course":"none","reading_score":72,"writing_score":74}'

# 5. Stop and remove the test container
docker rm -f sp-test