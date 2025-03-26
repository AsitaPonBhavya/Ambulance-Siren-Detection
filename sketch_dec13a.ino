const int relay1 = D1; // Relay connected to D1

void setup() {
  pinMode(relay1, OUTPUT);
  digitalWrite(relay1, LOW); // Red light ON by default
  Serial.begin(115200); // Initialize serial communication
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "NORMAL") {
      // Ambulance detected, give priority
      digitalWrite(relay1, LOW); // Green light ON

    }
    if (command == "AMBULANCE") {
      // Ambulance detected, give priority
      digitalWrite(relay1, HIGH); // Green light ON
      delay(10000); // Shift for 10 seconds
      digitalWrite(relay1, LOW); // Red light ON after priority time
    }
  }

  // Normal operation
  digitalWrite(relay1, HIGH);
  delay(3000); // Green light duration
  
  digitalWrite(relay1, LOW);
  delay(3000); // Red light duration
}