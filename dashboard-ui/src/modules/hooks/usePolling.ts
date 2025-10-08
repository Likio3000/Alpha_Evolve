import { useEffect, useRef } from "react";

export function usePolling(callback: () => void, delayMs: number | null, active: boolean): void {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!active || delayMs === null) {
      return;
    }

    const id = window.setInterval(() => {
      savedCallback.current();
    }, delayMs);

    return () => {
      window.clearInterval(id);
    };
  }, [delayMs, active]);
}
