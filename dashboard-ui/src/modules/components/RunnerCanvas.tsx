import React, { useEffect, useRef } from "react";

interface Particle {
  angle: number;
  baseRadius: number;
  speed: number;
  size: number;
  wobbleAmplitude: number;
  wobbleSpeed: number;
  pulseOffset: number;
}

function createParticle(maxRadius: number): Particle {
  const baseRadius = maxRadius * (0.35 + Math.random() * 0.55);
  return {
    angle: Math.random() * Math.PI * 2,
    baseRadius,
    speed: 0.25 + Math.random() * 0.75, // radians per second
    size: 1.6 + Math.random() * 2.8,
    wobbleAmplitude: baseRadius * (0.035 + Math.random() * 0.045),
    wobbleSpeed: 0.35 + Math.random() * 0.85,
    pulseOffset: Math.random() * Math.PI * 2,
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
    let lastFrameTime: number | null = null;
    let elapsed = 0;
    const particles: Particle[] = [];

    const resizeAndInit = () => {
      const { clientWidth, clientHeight } = canvas;
      const devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const renderWidth = Math.max(1, Math.floor(clientWidth * devicePixelRatio));
      const renderHeight = Math.max(1, Math.floor(clientHeight * devicePixelRatio));

      if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
        canvas.width = renderWidth;
        canvas.height = renderHeight;
      }
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

      const maxRadius = Math.max(clientWidth, clientHeight) * 0.65;
      particles.length = 0;
      const count = Math.max(24, Math.floor(maxRadius / 10));
      for (let i = 0; i < count; i += 1) {
        particles.push(createParticle(maxRadius));
      }
      lastFrameTime = null;
    };

    const draw = (timeSeconds: number) => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      const baseGradient = ctx.createRadialGradient(
        centerX,
        centerY,
        Math.min(width, height) * 0.15,
        centerX,
        centerY,
        Math.max(width, height) * 1.1,
      );
      baseGradient.addColorStop(0, "rgba(10, 24, 40, 0.9)");
      baseGradient.addColorStop(0.6, "rgba(8, 20, 32, 0.95)");
      baseGradient.addColorStop(1, "rgba(4, 10, 18, 0.98)");

      ctx.fillStyle = baseGradient;
      ctx.fillRect(0, 0, width, height);

      const sweepAngle = timeSeconds * 0.25;
      const highlightX = centerX + Math.cos(sweepAngle) * width * 0.25;
      const highlightY = centerY + Math.sin(sweepAngle) * height * 0.25;
      const highlightRadius = Math.max(width, height) * 0.8;
      const highlight = ctx.createRadialGradient(highlightX, highlightY, 0, highlightX, highlightY, highlightRadius);
      highlight.addColorStop(0, "rgba(70, 160, 220, 0.18)");
      highlight.addColorStop(1, "rgba(70, 160, 220, 0)");
      ctx.fillStyle = highlight;
      ctx.fillRect(0, 0, width, height);

      particles.forEach((particle, index) => {
        const wobble = Math.sin(timeSeconds * particle.wobbleSpeed + particle.pulseOffset) * particle.wobbleAmplitude;
        const radius = Math.max(12, particle.baseRadius + wobble);
        const x = centerX + Math.cos(particle.angle) * radius;
        const y = centerY + Math.sin(particle.angle) * radius * 0.65;

        const alpha = 0.22 + 0.58 * ((Math.sin(timeSeconds * 0.9 + particle.pulseOffset) + 1) / 2);
        const shadow = 6 + Math.sin(timeSeconds * 0.6 + index) * 3;

        ctx.beginPath();
        ctx.shadowBlur = shadow;
        ctx.shadowColor = "rgba(90, 180, 240, 0.38)";
        ctx.fillStyle = `rgba(90, 180, 240, ${alpha})`;
        ctx.arc(x, y, particle.size, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.shadowBlur = 0;
    };

    const update = (deltaSeconds: number) => {
      particles.forEach((particle) => {
        particle.angle += particle.speed * deltaSeconds;
        if (particle.angle > Math.PI * 2) {
          particle.angle -= Math.PI * 2;
        }
      });
    };

    const loop = (frameTime: number) => {
      if (lastFrameTime === null) {
        lastFrameTime = frameTime;
      }
      const delta = Math.min((frameTime - lastFrameTime) / 1000, 0.05);
      lastFrameTime = frameTime;
      elapsed += delta;

      update(delta);
      draw(elapsed);

      animationId = window.requestAnimationFrame(loop);
    };

    const handleResize = () => {
      resizeAndInit();
    };

    resizeAndInit();
    window.addEventListener("resize", handleResize);
    animationId = window.requestAnimationFrame(loop);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="runner-canvas" />;
}
