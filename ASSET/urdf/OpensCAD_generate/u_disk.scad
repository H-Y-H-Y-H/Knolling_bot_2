// Parameters for the USB Disk
length = 60; // Length of the USB Disk
width = 20;  // Width of the USB Disk
height = 10; // Height of the USB Disk

// Parameters for the USB Connector
conn_length = 70; // Length of the Connector
conn_width = 12;  // Width of the Connector
conn_height = 5;  // Height of the Connector

module usb_disk() {
    difference() {
        // Body of the USB Disk, centered around the origin
        translate([-length/2, -width/2, -height/2])
        cube([length, width, height]);

        // Hollow space for the USB Connector
        translate([-conn_length/2, -conn_width/2, -conn_height/2])
        cube([conn_length, conn_width, conn_height]);
    }
}

module usb_connector() {
    // USB Connector part, positioned relative to the disk body
    translate([length/2 - conn_length, -conn_width/2, -conn_height/2])
    cube([conn_length, conn_width, conn_height]);
}

// Combine the USB disk body and the connector
usb_disk();
usb_connector();
