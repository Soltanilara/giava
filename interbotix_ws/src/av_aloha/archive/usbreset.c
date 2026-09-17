/* usbreset -- send a USB port reset to a USB device

    1) Find the device: lsusb

    Example: Bus 001 Device 004: ID 8086:0b07 Intel RealSense

    2) Compose the device filename: /dev/bus/usb/BBB/DDD

    where BBB = zero-padded bus number and DDD = zero-padded device number

    Example: /dev/bus/usb/001/004

    3) Compile: gcc -o usbreset usbreset.c

    4) Run as root: sudo ./usbreset /dev/bus/usb/001/004

*/

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <linux/usbdevice_fs.h>

int main(int argc, char **argv)
{
    const char *filename;
    int fd;
    int rc;

    if (argc != 2) {
        fprintf(stderr, "Usage: usbreset device-filename\n");
        return 1;
    }
    filename = argv[1];

    fd = open(filename, O_WRONLY);
    if (fd < 0) {
        perror("Error opening output file");
        return 1;
    }

    printf("Resetting USB device %s\n", filename);
    rc = ioctl(fd, USBDEVFS_RESET, 0);
    if (rc < 0) {
        perror("Error in ioctl");
        return 1;
    }
    printf("Reset successful\n");

    close(fd);
    return 0;
}
