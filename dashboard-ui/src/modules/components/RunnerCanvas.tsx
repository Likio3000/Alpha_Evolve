import React, { useEffect, useRef } from "react";

interface Particle {
  angle: number;
  radius: number;
  speed: number;
  size: number;
}

function createParticle(maxRadius: number): Particle {
  return {
    angle: Math.random() * Math.PI * 2,
    radius: maxRadius * (0.2 + Math.random() * 0.8),
    speed: 0.001 + Math.random() * 0.0025,
    size: 1.5 + Math.random() * 2.5,
  };
}

export function RunnerCanvas(): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    let animationId = 0;
    const particles: Particle[] = [];

    const syncSize = () => {
      const { clientWidth, clientHeight } = canvas;
      canvas.width = clientWidth;
      canvas.height = clientHeight;
      const maxRadius = Math.max(clientWidth, clientHeight) * 0.5;
      particles.length = 0;
      const count = Math.floor(maxRadius / 8);
      for (let i = 0; i < count; i += 1) {
        particles.push(createParticle(maxRadius));
      }
    };

    const draw = (time: number) => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, Math.max(width, height));
      gradient.addColorStop(0, "rgba(18, 34, 50, 0.65)");
      gradient.addColorStop(1, "rgba(6, 10, 14, 0.95)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      particles.forEach((particle, index) => {
        const t = time * particle.speed;
        const wobble = Math.sin(t * 0.002 + index) * particle.radius * 0.02;
        const angle = particle.angle + t * 0.0018;
        const distance = particle.radius + wobble;
        const x = centerX + Math.cos(angle) * distance;
        const y = centerY + Math.sin(angle) * distance;

        const alpha = 0.15 + 0.75 * ((Math.sin(time * 0.001 + index) + 1) / 2);

        ctx.beginPath();
        ctx.fillStyle = `rgba(90, 180, 240, ${alpha})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(90, 180, 240, 0.35)";
        ctx.arc(x, y, particle.size, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    const loop = (time: number) => {
      draw(time);
      animationId = window.requestAnimationFrame(loop);
    };

    const handleResize = () => {
      syncSize();
    };

    syncSize();
    window.addEventListener("resize", handleResize);
    animationId = window.requestAnimationFrame(loop);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="runner-canvas" />;
}
