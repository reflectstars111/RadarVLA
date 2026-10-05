frame_t
│
├── sensors
│   ├── camera
│   │    ├── front_rgb
│   │    ├── left/right (optional)
│   │
│   ├── lidar
│   │    └── pointcloud
│   │
│   └── radar
│        ├── power
│        ├── folded_doppler
│        ├── unfolded_doppler
│        └── raw points/cube(optional)
│
├── ego
│   ├── pose
│   ├── velocity
│   ├── acceleration
│   └── yaw rate
│
├── agents
│   ├── id
│   ├── bbox3d
│   ├── velocity
│   ├── heading
│   └── future trajectory
│
├── map
│   ├── lane centerlines
│   ├── lane boundaries
│   └── traffic elements
│
└── language
    └── instruction
