import { renderHook, act } from "@testing-library/react";
import { usePolling } from "./usePolling";
describe("usePolling", () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });
    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
    });
    it("invokes callback at interval while active", () => {
        const callback = vi.fn();
        const { rerender, unmount } = renderHook(({ active }) => {
            usePolling(callback, 1000, active);
        }, { initialProps: { active: true } });
        act(() => {
            vi.advanceTimersByTime(1000);
        });
        expect(callback).toHaveBeenCalledTimes(1);
        act(() => {
            vi.advanceTimersByTime(2000);
        });
        expect(callback).toHaveBeenCalledTimes(3);
        rerender({ active: false });
        act(() => {
            vi.advanceTimersByTime(1000);
        });
        expect(callback).toHaveBeenCalledTimes(3);
        unmount();
    });
    it("does not start interval when delay is null", () => {
        const callback = vi.fn();
        renderHook(() => {
            usePolling(callback, null, true);
        });
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        expect(callback).not.toHaveBeenCalled();
    });
});
