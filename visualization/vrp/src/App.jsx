import { useState, useEffect, useRef, useMemo } from 'react'
import Map from 'react-map-gl/maplibre'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer, TextLayer, PathLayer } from '@deck.gl/layers'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

// OpenStreetMap tile style (free, no API key needed)
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'

function App() {
  const [data, setData] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(200)
  const [panelOpen, setPanelOpen] = useState(true)
  const canvasRef = useRef(null)

  // Load episode data
  useEffect(() => {
    fetch('/episode_data.json')
      .then(res => res.json())
      .then(d => {
        setData(d)
        setCurrentStep(0)
      })
      .catch(err => console.error('Failed to load data:', err))
  }, [])

  // Auto-play
  useEffect(() => {
    if (!isPlaying || !data) return
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= data.timesteps.length - 1) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, playSpeed)
    return () => clearInterval(interval)
  }, [isPlaying, data, playSpeed])

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

      // Current truck route (highlighted)
      const currentNode = timestep.truck.current_node
      const targetNode = timestep.truck.target_node
      if (currentNode !== null && targetNode !== null && currentNode !== targetNode) {
        const pathKey = `${currentNode}-${targetNode}`
        const currentPath = data.road_paths[pathKey]
        if (currentPath) {
          const currentRouteLayer = new PathLayer({
            id: 'current-route',
            data: [{ path: currentPath }],
            getPath: d => d.path,
            getColor: [59, 130, 246, 200],
            getWidth: 6,
            widthMinPixels: 4,
            pickable: false
          })
          layersList.push(currentRouteLayer)
        }
      }
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

    // Customer labels
    const customerLabelsLayer = new TextLayer({
      id: 'customer-labels',
      data: data.customers,
      getPosition: d => [d.lon, d.lat],
      getText: d => d.name || `C${d.id}`,
      getSize: 12,
      getColor: [255, 255, 255, 255],
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'center',
      fontFamily: 'Arial',
      fontWeight: 'bold',
      background: true,
      getBackgroundColor: [0, 0, 0, 150],
      backgroundPadding: [4, 2]
    })
    layersList.push(customerLabelsLayer)

    // Truck layer (blue)
    const truckLayer = new ScatterplotLayer({
      id: 'truck',
      data: [timestep.truck],
      getPosition: d => [d.lon, d.lat],
      getRadius: 60,
      getFillColor: [59, 130, 246, 255],
      pickable: true
    })
    layersList.push(truckLayer)

    // Truck label
    const truckLabelLayer = new TextLayer({
      id: 'truck-label',
      data: [timestep.truck],
      getPosition: d => [d.lon, d.lat],
      getText: () => 'Truck',
      getSize: 14,
      getColor: [255, 255, 255, 255],
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'center',
      fontWeight: 'bold'
    })
    layersList.push(truckLabelLayer)

    // Drones layer (colored circles)
    const droneColors = [
      [245, 158, 11, 255],  // orange
      [139, 92, 246, 255],  // purple
      [236, 72, 153, 255],  // pink
      [34, 197, 94, 255],   // green
      [14, 165, 233, 255]   // sky blue
    ]

    const dronesLayer = new ScatterplotLayer({
      id: 'drones',
      data: timestep.drones,
      getPosition: d => [d.lon, d.lat],
      getRadius: 35,
      getFillColor: d => droneColors[d.id % droneColors.length],
      pickable: true
    })
    layersList.push(dronesLayer)

    // Drone labels with battery
    const droneLabelLayer = new TextLayer({
      id: 'drone-labels',
      data: timestep.drones,
      getPosition: d => [d.lon, d.lat],
      getText: d => `D${d.id}\n${Math.round(d.battery * 100)}%`,
      getSize: 11,
      getColor: [255, 255, 255, 255],
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'bottom',
      fontWeight: 'bold',
      getPixelOffset: [0, -25]
    })
    layersList.push(droneLabelLayer)

    return layersList
  }, [data, currentStep])

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
            <Map mapStyle={MAP_STYLE} />
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
            <button onClick={() => setCurrentStep(0)} title="Reset">
              ⏮
            </button>
            <button onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} title="Previous">
              ◀
            </button>
            <button onClick={() => setIsPlaying(!isPlaying)} title={isPlaying ? 'Pause' : 'Play'}>
              {isPlaying ? '⏸' : '▶'}
            </button>
            <button onClick={() => setCurrentStep(Math.min(data.timesteps.length - 1, currentStep + 1))} title="Next">
              ▶
            </button>
            <button onClick={() => setCurrentStep(data.timesteps.length - 1)} title="End">
              ⏭
            </button>
          </div>
          <div className="slider-container">
            <input
              type="range"
              min={0}
              max={data.timesteps.length - 1}
              value={currentStep}
              onChange={e => setCurrentStep(Number(e.target.value))}
            />
          </div>
          <div className="speed-control">
            <label>Speed: {playSpeed}ms</label>
            <input
              type="range"
              min={50}
              max={500}
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
              <span className="line blue"></span>
              <span>Current Route</span>
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
