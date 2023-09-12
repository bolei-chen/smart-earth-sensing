class Instruction:
    def __init__(self, inst, start_time):
        self.cable_id = int(inst[0])
        self.starting_cumulative_time = time_to_sec(inst[1]) - start_time
        self.label = int(inst[2]) 
        self.duration = int(inst[3]) 
        self.ending_cumulative_time = self.starting_cumulative_time + self.duration

    def __str__(self):
        inst = "-------------------\n" 
        inst += "cable id: " + str(self.cable_id) + "\n"
        inst += "starting cumulative time: " + str(self.starting_cumulative_time) + "\n"
        inst += "ending cumulative time: " + str(self.ending_cumulative_time) + "\n"
        inst += "duration: " + str(self.duration) + "\n"
        inst += "class: " + str(self.label)
        return inst 
     
''' 
returns the total seconds 
Input: 
    time: a string which represent time in the form HH:MM:SS, for example: 14:20:30
''' 
def time_to_sec(time):
    sections = time.split(":")
    hours = float(sections[0])
    minutes = float(sections[1])
    seconds = float(sections[2])
    return int(3600 * hours + 60 * minutes + seconds)