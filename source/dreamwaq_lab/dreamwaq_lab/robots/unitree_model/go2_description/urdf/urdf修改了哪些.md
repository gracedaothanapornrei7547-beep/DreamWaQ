Open "go2_description.urdf" in "./go2_description/urdf",
and amend the ` box size="0.213 0.0245 0.034" ` in links of "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh".

For example, change previous values to ` box size="0.11 0.0245 0.034" ` means the length of the thigh is shortened from 0.213 to 0.11, which can avoid unnecessary collision between the thigh link and the calf link. 


1,thigh碰撞体改小一点。(0.15)
2,fixed joint 的 <axis xyz="0 0 0" />都删掉了
3,head_upper和head_lower加了visual
