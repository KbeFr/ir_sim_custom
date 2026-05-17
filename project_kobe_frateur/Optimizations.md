Things to optimize 

- In the simulation_gui a lot of seatching is done in the env_obstacle_list based on id, like this 
        obj = next((o for o in self.env.objects if o.id == oid), None)
If we make this into a dict with ids, would be faster

- Using sets instead of Lists on uniqe iterables, like in uavFleet for detected objects  

*TODO*
- Better Runtime UGV spawn in 
- Runtime mission removal option
- The local controllers of the UGV are managed by the simulation loop (gui).
This seems architecturally wrong since they should live in the UGV_DT