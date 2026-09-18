import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Eye, Layers, Maximize2, RotateCcw, AlertTriangle, CheckCircle2, Thermometer } from 'lucide-react';

/**
 * ThreeDHeatmap: 3D Server Room Digital Twin with 8x8 Rack Grid & 4 CRAC Towers.
 * Grounded in src/aws/twinmaker/scene_schema.json.
 */
export default function ThreeDHeatmap({ spatialGrid = [], onSelectRack, selectedRack }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const rackMeshesRef = useRef(new Map());
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());

  const [hoveredRack, setHoveredRack] = useState(null);
  const [activePreset, setActivePreset] = useState('iso');

  // Thermal colormap helper matching TwinMaker scene schema
  const getThermalColor = (tempC) => {
    if (tempC < 18.0) return new THREE.Color(0x00BFFF); // Blue
    if (tempC <= 25.0) {
      // Interpolate Blue-Green to Emerald
      const alpha = (tempC - 18.0) / 7.0;
      return new THREE.Color(0x00BFFF).lerp(new THREE.Color(0x10B981), alpha);
    }
    if (tempC <= 27.0) {
      // Interpolate Emerald to Amber
      const alpha = (tempC - 25.0) / 2.0;
      return new THREE.Color(0x10B981).lerp(new THREE.Color(0xF59E0B), alpha);
    }
    if (tempC <= 32.0) {
      // Interpolate Amber to Red
      const alpha = (tempC - 27.0) / 5.0;
      return new THREE.Color(0xF59E0B).lerp(new THREE.Color(0xEF4444), alpha);
    }
    return new THREE.Color(0xFF0000); // Critical Red
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 520;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x070B14);
    scene.fog = new THREE.FogExp2(0x070B14, 0.015);

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(24, 28, 38);
    cameraRef.current = camera;

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't go below floor
    controls.minDistance = 8;
    controls.maxDistance = 75;
    controls.target.set(0, 1.5, 0);
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0x1E293B, 2.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xE2E8F0, 1.8);
    dirLight.position.set(15, 25, 20);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    scene.add(dirLight);

    // Subtle blue accent light for tech atmosphere
    const bluePoint = new THREE.PointLight(0x06B6D4, 3.0, 40);
    bluePoint.position.set(-10, 8, -10);
    scene.add(bluePoint);

    // 6. Floor & Spatial Grid
    const floorGeo = new THREE.PlaneGeometry(36, 24);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x0F172A,
      roughness: 0.8,
      metalness: 0.2,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const gridHelper = new THREE.GridHelper(36, 36, 0x06B6D4, 0x1E293B);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    // 7. Create 8x8 Server Rack Grid (64 Racks)
    const rackWidth = 1.0;
    const rackHeight = 2.4;
    const rackDepth = 1.2;
    const rackGeo = new THREE.BoxGeometry(rackWidth, rackHeight, rackDepth);

    const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    const rackGroup = new THREE.Group();

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const rackId = `RACK-${rows[r]}${c < 9 ? '0' + (c + 1) : c + 1}`;
        // Spacing with aisle simulation
        const aisleSpacing = (r % 2 === 1 ? 0.8 : 0.4);
        const posX = (c - 3.5) * 1.8;
        const posZ = (r - 3.5) * 2.2 + aisleSpacing;

        const mat = new THREE.MeshStandardMaterial({
          color: 0x10B981,
          roughness: 0.4,
          metalness: 0.6,
        });

        const rackMesh = new THREE.Mesh(rackGeo, mat);
        rackMesh.position.set(posX, rackHeight / 2, posZ);
        rackMesh.castShadow = true;
        rackMesh.receiveShadow = true;
        rackMesh.userData = { rackId, row: r, col: c, temp_c: 22.5 };

        // Add rack front server grill mesh
        const grillGeo = new THREE.PlaneGeometry(rackWidth * 0.85, rackHeight * 0.85);
        const grillMat = new THREE.MeshBasicMaterial({ color: 0x0A0F1D });
        const grill = new THREE.Mesh(grillGeo, grillMat);
        grill.position.set(0, 0, rackDepth / 2 + 0.01);
        rackMesh.add(grill);

        // Blinking LED indicator
        const ledGeo = new THREE.SphereGeometry(0.04, 8, 8);
        const ledMat = new THREE.MeshBasicMaterial({ color: 0x00FF88 });
        const led = new THREE.Mesh(ledGeo, ledMat);
        led.position.set(rackWidth * 0.35, rackHeight * 0.35, rackDepth / 2 + 0.02);
        rackMesh.add(led);

        rackGroup.add(rackMesh);
        rackMeshesRef.current.set(rackId, rackMesh);
      }
    }
    scene.add(rackGroup);

    // 8. Create 4 CRAC Cooling Towers
    const cracGroup = new THREE.Group();
    const cracGeo = new THREE.BoxGeometry(2.0, 3.2, 1.6);
    const cracPositions = [
      { id: 'CRAC-01', x: -14, z: -8, label: 'North' },
      { id: 'CRAC-02', x: -14, z: 8,  label: 'South' },
      { id: 'CRAC-03', x: 14,  z: -8, label: 'East'  },
      { id: 'CRAC-04', x: 14,  z: 8,  label: 'West'  },
    ];

    cracPositions.forEach((crac) => {
      const cracMat = new THREE.MeshStandardMaterial({
        color: 0x1E3A8A, // Navy Blue
        roughness: 0.3,
        metalness: 0.7,
      });
      const cracMesh = new THREE.Mesh(cracGeo, cracMat);
      cracMesh.position.set(crac.x, 1.6, crac.z);
      cracMesh.castShadow = true;
      cracMesh.userData = { isCrac: true, cracId: crac.id, label: crac.label };

      // Cyan intake fan ring
      const ringGeo = new THREE.RingGeometry(0.3, 0.6, 16);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x00D4FF, side: THREE.DoubleSide });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.set(0, 0.4, 0.81);
      cracMesh.add(ring);

      cracGroup.add(cracMesh);
    });
    scene.add(cracGroup);

    // 9. Mouse Raycasting Handler
    const handlePointerMove = (e) => {
      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(rackGroup.children);

      if (intersects.length > 0) {
        const hit = intersects[0].object;
        if (hit.userData && hit.userData.rackId) {
          setHoveredRack(hit.userData);
          container.style.cursor = 'pointer';
          return;
        }
      }
      setHoveredRack(null);
      container.style.cursor = 'default';
    };

    const handleClick = (e) => {
      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(rackGroup.children);
      if (intersects.length > 0) {
        const hit = intersects[0].object;
        if (hit.userData && hit.userData.rackId && onSelectRack) {
          onSelectRack(hit.userData);
        }
      }
    };

    container.addEventListener('pointermove', handlePointerMove);
    container.addEventListener('click', handleClick);

    // 10. Animation Loop
    let animationFrameId;
    const clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Subtle breathing light on blue point
      bluePoint.intensity = 2.5 + Math.sin(elapsedTime * 2) * 0.5;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 11. Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 520;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      container.removeEventListener('pointermove', handlePointerMove);
      container.removeEventListener('click', handleClick);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [onSelectRack]);

  // Update rack colors dynamically when spatialGrid changes
  useEffect(() => {
    if (!spatialGrid || spatialGrid.length === 0) return;

    spatialGrid.forEach((item) => {
      const mesh = rackMeshesRef.current.get(item.rack_id);
      if (mesh) {
        const color = getThermalColor(item.temp_c);
        mesh.material.color.copy(color);
        mesh.userData = { ...mesh.userData, ...item };

        // Highlight selected rack with wireframe or emissive glow
        if (selectedRack && selectedRack.rack_id === item.rack_id) {
          mesh.material.emissive = new THREE.Color(0xFFFFFF);
          mesh.material.emissiveIntensity = 0.35;
        } else {
          mesh.material.emissive = new THREE.Color(0x000000);
          mesh.material.emissiveIntensity = 0;
        }
      }
    });
  }, [spatialGrid, selectedRack]);

  // Camera presets
  const applyCameraPreset = (preset) => {
    if (!cameraRef.current || !controlsRef.current) return;
    setActivePreset(preset);
    const cam = cameraRef.current;
    const ctrl = controlsRef.current;

    if (preset === 'iso') {
      cam.position.set(24, 28, 38);
      ctrl.target.set(0, 1.5, 0);
    } else if (preset === 'top') {
      cam.position.set(0, 42, 0.1);
      ctrl.target.set(0, 0, 0);
    } else if (preset === 'cold') {
      cam.position.set(0, 4, 12);
      ctrl.target.set(0, 2, 0);
    } else if (preset === 'crac') {
      cam.position.set(-20, 10, -10);
      ctrl.target.set(-14, 2, -8);
    }
    ctrl.update();
  };

  return (
    <div className="relative w-full h-[540px] rounded-2xl overflow-hidden glass-panel border border-slate-800 flex flex-col">
      {/* 3D Canvas Container */}
      <div ref={mountRef} className="w-full h-full" />

      {/* Top HUD Controls Overlay */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
        <div className="flex items-center space-x-3 pointer-events-auto bg-slate-950/80 backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-slate-800/80 text-xs">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-slate-200">8×8 SPATIAL TWIN</span>
          <span className="text-slate-500">|</span>
          <span className="text-emerald-400 font-mono">64 RACKS ACTIVE</span>
          <span className="text-slate-500">|</span>
          <span className="text-cyan-400 font-mono">4 CRAC ZONES</span>
        </div>

        {/* Camera View Switcher */}
        <div className="flex items-center space-x-1.5 pointer-events-auto bg-slate-950/80 backdrop-blur-md p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => applyCameraPreset('iso')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === 'iso' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Isometric
          </button>
          <button
            onClick={() => applyCameraPreset('top')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === 'top' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2D Heatmap
          </button>
          <button
            onClick={() => applyCameraPreset('cold')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === 'cold' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Aisle Walk
          </button>
          <button
            onClick={() => applyCameraPreset('crac')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activePreset === 'crac' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            CRAC North
          </button>
        </div>
      </div>

      {/* Hover / Tooltip Card */}
      {hoveredRack && (
        <div className="absolute bottom-16 left-4 pointer-events-none bg-slate-950/90 backdrop-blur-md p-3 rounded-xl border border-slate-700/80 shadow-2xl text-xs space-y-1.5 w-52 animate-fadeIn">
          <div className="flex items-center justify-between font-mono font-bold text-slate-100">
            <span>{hoveredRack.rackId}</span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] ${hoveredRack.temp_c > 27.0 ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
              {hoveredRack.temp_c > 27.0 ? 'SLA BREACH' : 'NORMAL'}
            </span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Inlet Temp:</span>
            <span className="font-mono text-slate-100 font-bold">{hoveredRack.temp_c}°C</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Power Load:</span>
            <span className="font-mono text-slate-100">{hoveredRack.power_kw || '24.2'} kW</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Zone CDU:</span>
            <span className="font-mono text-cyan-400">{hoveredRack.crac_id || 'CRAC-01'}</span>
          </div>
        </div>
      )}

      {/* Bottom Thermal Colormap Legend Bar */}
      <div className="absolute bottom-4 left-4 right-4 pointer-events-auto bg-slate-950/85 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-800/90 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2">
          <Thermometer className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-400 font-medium">ASHRAE TC 9.9 Thermal Spectrum:</span>
        </div>

        {/* Gradient Legend Track */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#00BFFF]" />
            <span className="text-slate-400 text-[11px]">&lt; 18°C (Cold)</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
            <span className="text-emerald-400 text-[11px] font-semibold">18–25°C (Optimal)</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B]" />
            <span className="text-amber-400 text-[11px]">25–27°C (Allowable)</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
            <span className="text-red-400 text-[11px] font-bold">&gt; 27°C (Breach)</span>
          </div>
        </div>

        <span className="text-[11px] text-slate-500 font-mono">Click rack to inspect telemetry</span>
      </div>
    </div>
  );
}
