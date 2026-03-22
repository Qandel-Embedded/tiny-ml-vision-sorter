/*
 * Raspberry Pi Pico (RP2040) + Arducam + TensorFlow Lite Micro
 * Vision Sorter: Classifies objects on a conveyor belt and actuates a servo.
 */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "image_provider.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"

#define SERVO_PIN 15
#define THRESHOLD_SCORE 0.8f

const int kTensorArenaSize = 136 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void set_servo_angle(int angle) {
    uint slice_num = pwm_gpio_to_slice_num(SERVO_PIN);
    // Convert 0-180 angle to pulse width 1000-2000 microsecs
    int pulse = 1000 + (angle * 1000 / 180);
    // Assuming 50Hz PWM config setup earlier
    pwm_set_chan_level(slice_num, pwm_gpio_to_channel(SERVO_PIN), pulse);
}

int main() {
    stdio_init_all();
    init_camera();
    
    // Setup PWM for Servo
    gpio_set_function(SERVO_PIN, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(SERVO_PIN);
    pwm_set_wrap(slice_num, 20000); // 20ms = 50Hz
    pwm_set_clkdiv(slice_num, 125.0f); // 1MHz clock
    pwm_set_enabled(slice_num, true);

    const tflite::Model* model = tflite::GetModel(g_model_data);
    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    while (true) {
        // Capture 96x96 image
        if (get_image(input->data.int8, 96, 96)) {
            interpreter.Invoke();

            float defect_score = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
            printf("Defect Score: %f\n", defect_score);

            if (defect_score > THRESHOLD_SCORE) {
                printf("Defect detected! Rejecting.\n");
                set_servo_angle(90);  // Reject bin
            } else {
                set_servo_angle(0);   // Keep lane
            }
        }
    }
    return 0;
}
