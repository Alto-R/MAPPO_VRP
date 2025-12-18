import { useState, useEffect, useRef, useMemo } from 'react'
import MapGL from 'react-map-gl/maplibre'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer, PathLayer } from '@deck.gl/layers'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

// OpenStreetMap tile style (free, no API key needed)
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'

function App() {
  const [data, setData] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(500)
  const [panelOpen, setPanelOpen] = useState(true)
  const canvasRef = useRef(null)

  // Complete truck trajectory (dense path from GraphHopper)
  const [truckFullPath, setTruckFullPath] = useState(null)
  const [truckPathIndex, setTruckPathIndex] = useState(0) // 主时间轴：货车在轨迹上的位置
  const [pathIndexToStep, setPathIndexToStep] = useState(null) // pathIndex -> step 的映射
  const [routesLoading, setRoutesLoading] = useState(false)

  // Fetch route from GraphHopper
  const fetchRoute = async (fromLon, fromLat, toLon, toLat) => {
    try {
      const url = `http://localhost:8990/route?point=${fromLat},${fromLon}&point=${toLat},${toLon}&profile=car&points_encoded=false`
      const response = await fetch(url)
      const data = await response.json()
      if (data.paths && data.paths[0] && data.paths[0].points) {
        return data.paths[0].points.coordinates // [[lon, lat], ...]
      }
    } catch (e) {
      console.warn('GraphHopper fetch failed:', e)
    }
    return null
  }

  // Load episode data
  useEffect(() => {
    fetch('/episode_data.json')
      .then(res => res.json())
      .then(d => {
        setData(d)
        setCurrentStep(0)
        setTruckFullPath(null)
      })
      .catch(err => console.error('Failed to load data:', err))
  }, [])

  // Build complete truck trajectory after data loads
  useEffect(() => {
    if (!data || !data.has_geo || truckFullPath || routesLoading) return

    const buildTruckTrajectory = async () => {
      setRoutesLoading(true)
      console.log('Building complete truck trajectory...')

      const timesteps = data.timesteps

      // 提取货车移动的关键点，记录每个点对应的 stepIndex
      const waypoints = []
      let lastLon = null, lastLat = null

      for (let i = 0; i < timesteps.length; i++) {
        const truck = timesteps[i].truck
        const dist = lastLon !== null
          ? Math.sqrt((truck.lon - lastLon) ** 2 + (truck.lat - lastLat) ** 2)
          : Infinity

        if (dist > 0.00001) {
          waypoints.push({ lon: truck.lon, lat: truck.lat, stepIndex: i })
          lastLon = truck.lon
          lastLat = truck.lat
        }
      }

      console.log(`Found ${waypoints.length} unique truck waypoints`)

      // 构建完整轨迹，同时记录每个路径点对应的 step（带小数进度）
      const fullPath = []
      const pathToStep = [] // pathToStep[pathIdx] = { step, progress }

      for (let i = 0; i < waypoints.length - 1; i++) {
        const from = waypoints[i]
        const to = waypoints[i + 1]

        // 获取 GraphHopper 路线
        const routePath = await fetchRoute(from.lon, from.lat, to.lon, to.lat)

        let segmentPoints = []
        if (routePath && routePath.length > 0) {
          const startIdx = fullPath.length > 0 ? 1 : 0
          for (let j = startIdx; j < routePath.length; j++) {
            segmentPoints.push(routePath[j])
          }
        } else {
          if (fullPath.length === 0) {
            segmentPoints.push([from.lon, from.lat])
          }
          segmentPoints.push([to.lon, to.lat])
        }

        // 为这段路径的每个点分配 step 和 progress
        // 从 from.stepIndex 线性过渡到 to.stepIndex
        const fromStep = from.stepIndex
        const toStep = to.stepIndex
        const numPoints = segmentPoints.length

        for (let j = 0; j < numPoints; j++) {
          fullPath.push(segmentPoints[j])
          // 线性插值：pathProgress 从 0 到 1
          const pathProgress = numPoints > 1 ? j / (numPoints - 1) : 1
          // 对应的 step（可以是小数）
          const exactStep = fromStep + pathProgress * (toStep - fromStep)
          const step = Math.floor(exactStep)
          const progress = exactStep - step
          pathToStep.push({ step, progress })
        }
      }

      // 处理没有移动的情况
      if (fullPath.length === 0 && waypoints.length > 0) {
        fullPath.push([waypoints[0].lon, waypoints[0].lat])
        pathToStep.push({ step: 0, progress: 0 })
      }

      console.log(`Complete truck trajectory: ${fullPath.length} points`)
      console.log('PathToStep sample:', pathToStep.slice(0, 10))

      setTruckFullPath(fullPath)
      setPathIndexToStep(pathToStep)
      setRoutesLoading(false)
    }

    buildTruckTrajectory()
  }, [data, truckFullPath, routesLoading])

  // 根据 truckPathIndex 直接获取对应的 step 和 progress
  const getStepAndProgress = (pathIdx) => {
    if (!pathIndexToStep || pathIdx >= pathIndexToStep.length) {
      return { step: 0, progress: 0 }
    }
    return pathIndexToStep[pathIdx]
  }

  // 插值计算无人机位置
  const interpolateDronePosition = (droneIdx, step, progress) => {
    if (!data || step >= data.timesteps.length - 1) {
      const drone = data.timesteps[step].drones[droneIdx]
      return [drone.lon, drone.lat]
    }

    const currentDrone = data.timesteps[step].drones[droneIdx]
    const nextDrone = data.timesteps[step + 1].drones[droneIdx]

    // 线性插值
    const lon = currentDrone.lon + (nextDrone.lon - currentDrone.lon) * progress
    const lat = currentDrone.lat + (nextDrone.lat - currentDrone.lat) * progress
    return [lon, lat]
  }

  // Playback: 以货车轨迹为主时间轴
  useEffect(() => {
    if (!isPlaying || !data) return

    if (truckFullPath && truckFullPath.length > 1) {
      const totalPathPoints = truckFullPath.length

      const timer = setInterval(() => {
        setTruckPathIndex(prev => {
          if (prev >= totalPathPoints - 1) {
            setIsPlaying(false)
            return prev
          }
          return Math.min(prev + 1, totalPathPoints - 1)
        })
      }, playSpeed / 10) // 货车平滑移动

      return () => clearInterval(timer)
    } else {
      // 没有轨迹时的简单播放
      const timer = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= data.timesteps.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, playSpeed)

      return () => clearInterval(timer)
    }
  }, [isPlaying, data, playSpeed, truckFullPath])

  // 根据 truckPathIndex 更新 currentStep (用于 UI 显示)
  useEffect(() => {
    if (truckFullPath && pathIndexToStep) {
      const { step } = getStepAndProgress(truckPathIndex)
      setCurrentStep(step)
    }
  }, [truckPathIndex, truckFullPath, pathIndexToStep])

  // Helper: set step and sync truckPathIndex
  const setStepWithSync = (newStep) => {
    const clampedStep = Math.max(0, Math.min(newStep, data ? data.timesteps.length - 1 : 0))
    setCurrentStep(clampedStep)
    if (truckFullPath && pathIndexToStep) {
      // 找到第一个对应该 step 的路径点
      const pathIdx = pathIndexToStep.findIndex(p => p.step === clampedStep)
      if (pathIdx >= 0) {
        setTruckPathIndex(pathIdx)
      }
    }
  }

  // Calculate initial view state for the map
  const initialViewState = useMemo(() => {
    if (!data || !data.has_geo) return null
    const bounds = data.geo_bounds
    return {
      longitude: (bounds.min_lon + bounds.max_lon) / 2,
      latitude: (bounds.min_lat + bounds.max_lat) / 2,
      zoom: 13,
      pitch: 0,
      bearing: 0
    }
  }, [data])

  // Create deck.gl layers
  const layers = useMemo(() => {
    if (!data || !data.has_geo) return []

    const timestep = data.timesteps[currentStep]
    const layersList = []

    // Road paths layer (all paths, faded)
    if (data.road_paths && Object.keys(data.road_paths).length > 0) {
      const allPaths = Object.entries(data.road_paths).map(([key, path]) => ({
        id: key,
        path: path
      }))

      const roadPathsLayer = new PathLayer({
        id: 'road-paths-all',
        data: allPaths,
        getPath: d => d.path,
        getColor: [150, 150, 150, 80],
        getWidth: 3,
        widthMinPixels: 2,
        pickable: false
      })
      layersList.push(roadPathsLayer)
    }

    // Complete truck trajectory layer (blue line)
    if (truckFullPath && truckFullPath.length > 1) {
      const truckPathLayer = new PathLayer({
        id: 'truck-trajectory',
        data: [{ path: truckFullPath }],
        getPath: d => d.path,
        getColor: [59, 130, 246, 200],
        getWidth: 6,
        widthMinPixels: 3,
        pickable: false
      })
      layersList.push(truckPathLayer)
    }

    // Route nodes layer (gray circles)
    const routeNodesLayer = new ScatterplotLayer({
      id: 'route-nodes',
      data: data.route_nodes,
      getPosition: d => [d.lon, d.lat],
      getRadius: 30,
      getFillColor: [100, 100, 100, 200],
      pickable: true
    })
    layersList.push(routeNodesLayer)

    // Customers layer
    const customersLayer = new ScatterplotLayer({
      id: 'customers',
      data: data.customers.map((c, i) => ({
        ...c,
        served: timestep.customers_served.includes(i)
      })),
      getPosition: d => [d.lon, d.lat],
      getRadius: 40,
      getFillColor: d => d.served ? [74, 222, 128, 255] : [239, 68, 68, 255],
      pickable: true
    })
    layersList.push(customersLayer)

    // Truck layer (blue circle - follows the trajectory path)
    let truckPos
    if (truckFullPath && truckFullPath.length > 1) {
      // Use truckPathIndex directly to get position on the trajectory
      const pathIdx = Math.min(truckPathIndex, truckFullPath.length - 1)
      truckPos = truckFullPath[pathIdx]
    } else {
      truckPos = [timestep.truck.lon, timestep.truck.lat]
    }
    const truckLayer = new ScatterplotLayer({
      id: 'truck',
      data: [{ pos: truckPos }],
      getPosition: d => d.pos,
      getRadius: 120,
      getFillColor: [59, 130, 246, 255],
      pickable: true,
      transitions: {
        getPosition: {
          duration: playSpeed / 10, // Match the fast update interval
          easing: t => t
        }
      }
    })
    layersList.push(truckLayer)

    // Drones layer (colored circles) - 使用插值位置
    const droneColors = [
      [245, 158, 11, 255],  // orange
      [139, 92, 246, 255],  // purple
      [236, 72, 153, 255],  // pink
      [34, 197, 94, 255],   // green
      [14, 165, 233, 255]   // sky blue
    ]

    // 计算当前的 step 和进度用于无人机插值
    const { step: interpStep, progress: interpProgress } = truckFullPath && pathIndexToStep
      ? getStepAndProgress(truckPathIndex)
      : { step: currentStep, progress: 0 }

    const droneData = timestep.drones.map((drone, idx) => {
      // 使用插值计算无人机位置，与货车时间轴对齐
      const pos = truckFullPath && pathIndexToStep
        ? interpolateDronePosition(idx, interpStep, interpProgress)
        : [drone.lon, drone.lat]
      return {
        ...drone,
        pos
      }
    })

    const dronesLayer = new ScatterplotLayer({
      id: 'drones',
      data: droneData,
      getPosition: d => d.pos,
      getRadius: 100,
      getFillColor: d => droneColors[d.id % droneColors.length],
      pickable: true,
      // 货车和无人机使用相同的过渡时间
      transitions: {
        getPosition: {
          duration: playSpeed / 10,
          easing: t => t
        }
      }
    })
    layersList.push(dronesLayer)

    return layersList
  }, [data, currentStep, truckFullPath, truckPathIndex, pathIndexToStep, playSpeed])

  // Canvas-based rendering for non-geo data
  useEffect(() => {
    if (!data || data.has_geo || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // Clear
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    // Convert coordinates (-1, 1) to canvas
    const toCanvasX = x => (x + 1) * width / 2
    const toCanvasY = y => (1 - y) * height / 2  // Flip Y

    const timestep = data.timesteps[currentStep]

    // Draw route nodes (gray circles)
    ctx.fillStyle = '#444'
    data.route_nodes.forEach(node => {
      ctx.beginPath()
      ctx.arc(toCanvasX(node.x), toCanvasY(node.y), 8, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw customers
    data.customers.forEach((customer, i) => {
      const served = timestep.customers_served.includes(i)
      ctx.fillStyle = served ? '#4ade80' : '#ef4444'
      ctx.beginPath()
      ctx.arc(toCanvasX(customer.x), toCanvasY(customer.y), 12, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(`C${i}`, toCanvasX(customer.x), toCanvasY(customer.y) + 4)
    })

    // Draw truck (blue square)
    const truck = timestep.truck
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(toCanvasX(truck.x) - 15, toCanvasY(truck.y) - 10, 30, 20)
    ctx.fillStyle = '#fff'
    ctx.font = '10px Arial'
    ctx.textAlign = 'center'
    ctx.fillText('Truck', toCanvasX(truck.x), toCanvasY(truck.y) + 4)

    // Draw drones
    const droneColors = ['#f59e0b', '#8b5cf6', '#ec4899']
    timestep.drones.forEach((drone, i) => {
      ctx.fillStyle = droneColors[i % droneColors.length]
      ctx.beginPath()
      ctx.moveTo(toCanvasX(drone.x), toCanvasY(drone.y) - 10)
      ctx.lineTo(toCanvasX(drone.x) - 8, toCanvasY(drone.y) + 8)
      ctx.lineTo(toCanvasX(drone.x) + 8, toCanvasY(drone.y) + 8)
      ctx.closePath()
      ctx.fill()
      ctx.fillStyle = drone.battery > 0.3 ? '#4ade80' : '#ef4444'
      ctx.fillRect(toCanvasX(drone.x) - 15, toCanvasY(drone.y) + 12, 30 * drone.battery, 4)
      ctx.strokeStyle = '#888'
      ctx.strokeRect(toCanvasX(drone.x) - 15, toCanvasY(drone.y) + 12, 30, 4)
    })

  }, [data, currentStep])

  if (!data) {
    return (
      <div className="app">
        <div className="loading">
          <h1>VRP MAPPO Visualization</h1>
          <p>Loading episode data...</p>
          <p className="hint">
            Run <code>python mappo/scripts/run_demo.py</code> to generate data
          </p>
        </div>
      </div>
    )
  }

  const timestep = data.timesteps[currentStep]
  const routeStatus = routesLoading
    ? 'Loading trajectory...'
    : (truckFullPath ? `${truckFullPath.length} points` : 'No trajectory')

  return (
    <div className="app">
      {/* Full screen map */}
      <div className="map-container">
        {data.has_geo ? (
          <DeckGL
            initialViewState={initialViewState}
            controller={true}
            layers={layers}
            style={{ width: '100%', height: '100%' }}
          >
            <MapGL mapStyle={MAP_STYLE} />
          </DeckGL>
        ) : (
          <canvas ref={canvasRef} width={window.innerWidth} height={window.innerHeight} />
        )}
      </div>

      {/* Toggle button */}
      <button
        className="panel-toggle"
        onClick={() => setPanelOpen(!panelOpen)}
      >
        {panelOpen ? '◀' : '▶'}
      </button>

      {/* Side panel */}
      <div className={`side-panel ${panelOpen ? 'open' : 'closed'}`}>
        <div className="panel-header">
          <h2>VRP MAPPO {data.has_geo && '- Shenzhen'}</h2>
        </div>

        <div className="panel-section">
          <h3>Playback</h3>
          <div className="step-display">
            Step: {currentStep} / {data.timesteps.length - 1}
          </div>
          <div className="controls">
            <button onClick={() => setStepWithSync(0)} title="Reset">
              ⏮
            </button>
            <button onClick={() => setStepWithSync(currentStep - 1)} title="Previous">
              ◀
            </button>
            <button onClick={() => setIsPlaying(!isPlaying)} title={isPlaying ? 'Pause' : 'Play'}>
              {isPlaying ? '⏸' : '▶'}
            </button>
            <button onClick={() => setStepWithSync(currentStep + 1)} title="Next">
              ▶
            </button>
            <button onClick={() => setStepWithSync(data.timesteps.length - 1)} title="End">
              ⏭
            </button>
          </div>
          <div className="slider-container">
            <input
              type="range"
              min={0}
              max={data.timesteps.length - 1}
              value={currentStep}
              onChange={e => setStepWithSync(Number(e.target.value))}
            />
          </div>
          <div className="speed-control">
            <label>Speed: {playSpeed}ms</label>
            <input
              type="range"
              min={50}
              max={1000}
              step={50}
              value={playSpeed}
              onChange={e => setPlaySpeed(Number(e.target.value))}
            />
          </div>
        </div>

        <div className="panel-section">
          <h3>Episode Info</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Drones</span>
              <span className="info-value">{data.config.num_drones}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Customers</span>
              <span className="info-value">{data.config.num_customers}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Steps</span>
              <span className="info-value">{data.summary.total_steps}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Reward</span>
              <span className="info-value">{data.summary.total_reward.toFixed(1)}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Served</span>
              <span className="info-value">{data.summary.customers_served}/{data.summary.total_customers}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Step Reward</span>
              <span className="info-value">{timestep.reward.toFixed(2)}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Truck Path</span>
              <span className="info-value" style={{ fontSize: '12px' }}>{routeStatus}</span>
            </div>
          </div>
        </div>

        <div className="panel-section">
          <h3>Current State</h3>
          <div className="state-details">
            <div className="agent-state truck">
              <span className="agent-icon">🚚</span>
              <span className="agent-name">Truck</span>
              <span className="agent-pos">
                {data.has_geo
                  ? `${timestep.truck.lon?.toFixed(4)}, ${timestep.truck.lat?.toFixed(4)}`
                  : `${timestep.truck.x.toFixed(2)}, ${timestep.truck.y.toFixed(2)}`
                }
              </span>
            </div>
            {timestep.drones.map((d, i) => (
              <div key={i} className={`agent-state drone drone-${i}`}>
                <span className="agent-icon">🛸</span>
                <span className="agent-name">Drone {i}</span>
                <span className="agent-battery" style={{
                  color: d.battery > 0.3 ? '#4ade80' : '#ef4444'
                }}>
                  {(d.battery * 100).toFixed(0)}%
                </span>
                <span className="agent-status">{d.status}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel-section">
          <h3>Legend</h3>
          <div className="legend">
            <div className="legend-item">
              <span className="dot blue"></span>
              <span>Truck</span>
            </div>
            <div className="legend-item">
              <span className="line blue"></span>
              <span>Truck Trajectory</span>
            </div>
            <div className="legend-item">
              <span className="dot orange"></span>
              <span>Drone</span>
            </div>
            <div className="legend-item">
              <span className="dot red"></span>
              <span>Customer (waiting)</span>
            </div>
            <div className="legend-item">
              <span className="dot green"></span>
              <span>Customer (served)</span>
            </div>
            <div className="legend-item">
              <span className="dot gray"></span>
              <span>Route Node</span>
            </div>
            <div className="legend-item">
              <span className="line gray"></span>
              <span>Road Network</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
