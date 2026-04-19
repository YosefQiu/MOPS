/**
 * MOPS Pathline Data Loader and Visualization
 *
 * Loads pathline trajectory data from binary or JSON format
 * and provides rendering capabilities for canvas or WebGL.
 */

/**
 * Load pathlines from JSON format (easier for development/debugging)
 */
export async function loadPathlinesJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load pathlines: ${response.statusText}`);
  }
  const data = await response.json();
  return {
    format: data.format || 'unknown',
    numParticles: data.num_particles || data.particles.length,
    particles: data.particles
  };
}

/**
 * Load pathlines from binary format (more efficient for production)
 */
export async function loadPathlinesBinary(binUrl, metaUrl) {
  // Load metadata
  const metaResponse = await fetch(metaUrl);
  if (!metaResponse.ok) {
    throw new Error(`Failed to load metadata: ${metaResponse.statusText}`);
  }
  const metadata = await metaResponse.json();

  // Load binary data
  const binResponse = await fetch(binUrl);
  if (!binResponse.ok) {
    throw new Error(`Failed to load binary: ${binResponse.statusText}`);
  }
  const arrayBuffer = await binResponse.arrayBuffer();
  const dataView = new DataView(arrayBuffer);

  // Parse binary format
  let offset = 0;
  const numParticles = dataView.getInt32(offset, true); // little-endian
  offset += 4;

  const particles = [];
  const fieldsPerPoint = metadata.fields.length;
  const hasVelocity = metadata.fields.includes('velocity_u');
  const hasScalars = metadata.fields.includes('temperature');

  for (let i = 0; i < numParticles; i++) {
    const numPoints = dataView.getInt32(offset, true);
    offset += 4;

    if (numPoints === 0) {
      particles.push({ id: i, points: [] });
      continue;
    }

    const points = [];
    const velocity = hasVelocity ? [] : undefined;
    const temperature = hasScalars ? [] : undefined;
    const salinity = hasScalars ? [] : undefined;

    for (let j = 0; j < numPoints; j++) {
      const lat = dataView.getFloat64(offset, true);
      offset += 8;
      const lon = dataView.getFloat64(offset, true);
      offset += 8;

      points.push([lat, lon]);

      if (hasVelocity) {
        const u = dataView.getFloat64(offset, true);
        offset += 8;
        const v = dataView.getFloat64(offset, true);
        offset += 8;
        const speed = dataView.getFloat64(offset, true);
        offset += 8;
        velocity.push([u, v, speed]);
      }

      if (hasScalars) {
        const temp = dataView.getFloat64(offset, true);
        offset += 8;
        const sali = dataView.getFloat64(offset, true);
        offset += 8;
        temperature.push(temp);
        salinity.push(sali);
      }
    }

    const particle = { id: i, points };
    if (velocity) particle.velocity = velocity;
    if (temperature) particle.temperature = temperature;
    if (salinity) particle.salinity = salinity;

    particles.push(particle);
  }

  return {
    format: 'mops-pathlines-binary-v1',
    numParticles,
    particles,
    metadata
  };
}

/**
 * Render pathlines on a 2D canvas
 */
export function renderPathlines(canvas, pathlineData, options = {}) {
  const {
    projection = 'equirectangular',
    colorBy = 'particle', // 'particle', 'speed', 'temperature'
    lineWidth = 1.5,
    colormap = 'viridis',
    extent = null, // [lonMin, lonMax, latMin, latMax]
    backgroundColor = '#1a1a2e'
  } = options;

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, width, height);

  // Determine projection bounds
  const bounds = extent || [-180, 180, -90, 90];
  const [lonMin, lonMax, latMin, latMax] = bounds;

  // Convert lat/lon to canvas coordinates
  function projectToCanvas(lat, lon) {
    const x = ((lon - lonMin) / (lonMax - lonMin)) * width;
    const y = ((latMax - lat) / (latMax - latMin)) * height;
    return [x, y];
  }

  // Get color for particle
  function getColor(particleIdx, pointIdx, particle) {
    if (colorBy === 'particle') {
      const hue = (particleIdx / pathlineData.numParticles) * 360;
      return `hsl(${hue}, 70%, 60%)`;
    } else if (colorBy === 'speed' && particle.velocity) {
      const speed = particle.velocity[pointIdx]?.[2] || 0;
      const normalized = Math.min(speed / 2.0, 1.0); // Normalize to [0,1]
      return viridisColor(normalized);
    } else if (colorBy === 'temperature' && particle.temperature) {
      const temp = particle.temperature[pointIdx] || 0;
      const normalized = (temp - 0) / 30; // Assume 0-30°C range
      return viridisColor(normalized);
    }
    return '#00ff88';
  }

  // Draw each particle trajectory
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let i = 0; i < pathlineData.particles.length; i++) {
    const particle = pathlineData.particles[i];
    if (!particle.points || particle.points.length < 2) continue;

    ctx.beginPath();
    let penUp = true;

    for (let j = 0; j < particle.points.length; j++) {
      const [lat, lon] = particle.points[j];

      // Skip invalid points
      if (!isFinite(lat) || !isFinite(lon)) {
        penUp = true;
        continue;
      }

      const [x, y] = projectToCanvas(lat, lon);

      // Handle wrapping around date line
      if (j > 0) {
        const [prevLat, prevLon] = particle.points[j - 1];
        const lonDiff = Math.abs(lon - prevLon);
        if (lonDiff > 180) {
          penUp = true;
        }
      }

      if (penUp) {
        ctx.moveTo(x, y);
        penUp = false;
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.strokeStyle = getColor(i, 0, particle);
    ctx.stroke();
  }

  // Draw start points
  ctx.fillStyle = '#00ff88';
  for (const particle of pathlineData.particles) {
    if (particle.points.length > 0) {
      const [lat, lon] = particle.points[0];
      if (isFinite(lat) && isFinite(lon)) {
        const [x, y] = projectToCanvas(lat, lon);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

/**
 * Simple viridis colormap approximation
 */
function viridisColor(t) {
  t = Math.max(0, Math.min(1, t));

  const colors = [
    [68, 1, 84],      // 0.0 - purple
    [59, 82, 139],    // 0.25 - blue
    [33, 145, 140],   // 0.5 - teal
    [94, 201, 98],    // 0.75 - green
    [253, 231, 37]    // 1.0 - yellow
  ];

  const idx = t * (colors.length - 1);
  const i0 = Math.floor(idx);
  const i1 = Math.min(i0 + 1, colors.length - 1);
  const frac = idx - i0;

  const c0 = colors[i0];
  const c1 = colors[i1];

  const r = Math.round(c0[0] + (c1[0] - c0[0]) * frac);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * frac);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * frac);

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Export pathline data in fluid-earth compatible format
 * Note: This is for reference - fluid-earth in iframe cannot be modified directly
 */
export function convertToFluidEarthFormat(pathlineData) {
  // Fluid-earth uses a particle data format similar to:
  // Array of {x, y, vx, vy} where x,y are normalized screen coords
  // Since we can't inject into iframe, this is for local implementation

  const particles = [];

  for (const particle of pathlineData.particles) {
    if (particle.points.length === 0) continue;

    const trajectory = particle.points.map((point, idx) => {
      const [lat, lon] = point;
      const velocity = particle.velocity?.[idx] || [0, 0, 0];

      return {
        lat,
        lon,
        u: velocity[0],
        v: velocity[1],
        speed: velocity[2]
      };
    });

    particles.push({
      id: particle.id,
      trajectory,
      current: 0  // Current position index in trajectory
    });
  }

  return particles;
}

export default {
  loadPathlinesJSON,
  loadPathlinesBinary,
  renderPathlines,
  convertToFluidEarthFormat
};
