from threading import Thread
from time import sleep

import gpiozero
from gpiozero import LED


class light:
    def __init__(self):
        blueLED: LED = LED(18)
        yellowLED: LED = LED(25)
        greenLED: LED = LED(20)
        redLED: LED = LED(21)
        self.LEDMap = {
            "blue": blueLED,
            "yellow": yellowLED,
            "green": greenLED,
            "red": redLED,
            "None": LED(26),
        }

        self.classLightMap = {
            "hazardous": "red",
            "compost": "green",
            "garbage": "yellow",
            "recycle": "blue",
            "not-trash": "None",
        }

        Thread(target=self.setupFlash).start()

    def setupFlash(self):
        self.clearLights()
        for val in self.LEDMap.values():
            val.on()
            sleep(0.25)
            val.off()
        rev = list(self.LEDMap.values())
        rev.reverse()
        for val in rev:
            val.on()
            sleep(0.25)
            val.off()

    def clearLights(self):
        for i in self.LEDMap.values():
            try:
                i.off()
            except gpiozero.exc.GPIODeviceClosed:
                pass

    def lightUp(self, obj: str) -> None:
        Thread(target=self.clearLights).start()
        self.LEDMap[
            self.classLightMap[
                obj
            ]
        ].on()

    def __del__(self):
        self.clearLights()
        pass