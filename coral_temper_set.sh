echo "checking the temp of device"
cat /sys/class/thermal/thermal_zone0/temp

echo "Origional trip temp of device"
cat /sys/class/thermal/thermal_zone0/trip_point_4_temp 

sudo chmod 777 /sys/class/thermal/thermal_zone0/trip_point_4_temp 

cp /sys/class/thermal/thermal_zone0/trip_point_4_temp ~
sudo sed -i 's/65000/50000/g' ~/trip_point_4_temp
cp ~/trip_point_4_temp /sys/class/thermal/thermal_zone0/

echo "changed trip temperature"
cat /sys/class/thermal/thermal_zone0/trip_point_4_temp  

rm ~/trip_point_4_temp
