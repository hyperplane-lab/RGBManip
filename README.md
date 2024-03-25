# Adapose

To run experiments, use command line options: `python train.py dataset=?? task=?? pose_estimator=?? manipulation=?? controller=?? train=??`

Possible options include:

- dataset:
  - cabinet_train
  - cabinet_test
  - drawer_train
  - drawer_test
  - mug_train
  - mug_test
  - pot_train
  - pot_test
  - real_world
- task:
  - open_cabinet
  - open_cabinet_45
  - open_cabinet_no_dr
  - open_drawer
  - open_drawer_30
  - open_drawer_no_dr
  - open_pot
  - pick_mig
  - real_world
- pose_extimator:
  - adapose_cabinet
  - adapose_cabinet_baseline
  - adapose_drawer
  - adapose_drawer_baseline
  - adapose_mug
  - adapose_pot
  - ground_truth
- manipulation:
  - open_cabinet
  - open_cabinet_open_loop
  - open_drawer
  - open_drawer_open_looop
  - open_pot
  - open_pot_open_loop
  - pick_mug
  - pick_mug_open_loop
- controller
  - gt_pose
  - heuristic_pose
  - rl
- train
  - controller
  - test
  - collect: 采集离线数据集
  - test_baselines: 测试离线数据

Some other possible arguments are:

- task.num_envs=X: 并行环境数量
- headless=True/False: 是否显示图形界面
- viewerless=True/False: 是否需要渲染机械臂上的相机，用于加快不需要渲染的任务
- exp_name=XXX: 实验名称
- controller.load=Checkpoint: 导入保存的模型
