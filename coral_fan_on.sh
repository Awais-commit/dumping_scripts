sudo chmod 777 /sys/devices/virtual/thermal/thermal_zone0/mode
echo "disabled" > /sys/devices/virtual/thermal/thermal_zone0/mode

sudo chmod 777 /sys/devices/platform/gpio_fan/hwmon/hwmon0/fan1_target
echo 8600 > /sys/devices/platform/gpio_fan/hwmon/hwmon0/fan1_target
