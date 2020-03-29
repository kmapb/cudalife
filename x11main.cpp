#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

#include "gpulife.hpp"

void redraw(GPULife* gpulife, Display* dis, GC gc, int win){
  auto dims = gpulife->dims();
  const auto *cells = gpulife->cells();

  for (auto x = 0; x < dims.x; x++) {
    for (auto y = 0; y < dims.y; y++) {
      auto off = x * dims.y + y;
      if (cells[off]){
        XClearArea(dis, win, x, y, 1, 1, False);
      } else {
        XFillRectangle(dis, win, gc, x, y, 1,1);
      }
    }
  }
  // XSync(dis, False);
}

void handleEvent(GPULife* g, const XEvent* ev){
}

void drive(GPULife* gpulife){
  auto dims = gpulife->dims();

  Display *dis = XOpenDisplay(NULL);
  int screen = XDefaultScreen(dis);
  auto black = BlackPixel(dis, screen);
  auto white = WhitePixel(dis, screen);

  Window win = XCreateSimpleWindow(dis, DefaultRootWindow(dis),
        0, 0,
        dims.x, dims.y, 5, white, black);
  XSetStandardProperties(dis, win, "GPULife", "Oy!", None, NULL, 0, NULL);
  XSelectInput(dis, win,         
        ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
        ButtonPressMask | ButtonReleaseMask  | StructureNotifyMask );

  XMapWindow(dis, win);
  XFlush(dis);
  auto gc = XCreateGC(dis, win, 0, 0);
  XSetBackground(dis, gc, white);
  XSetBackground(dis, gc, white);
  XClearWindow(dis, win);
  XMapRaised(dis, win);

  // auto pixmap = XCreatePixmap(dis, win, dims.x, dims.y, 1);

  auto x11fd = ConnectionNumber(dis);
  for (;;) {
    gpulife->gen();

#if 1
    // Weird hackery to emulate non-blocking XNextEvent
    fd_set in_fds;
    FD_ZERO(&in_fds);
    FD_SET(x11fd, &in_fds);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    auto nready = select(x11fd + 1, &in_fds, NULL, NULL, &tv);
    if (nready > 0) {
      XEvent ev;
      XNextEvent(dis, &ev);
      handleEvent(gpulife, &ev);
    }
#endif

    redraw(gpulife, dis, gc, win);
    // XXX put it in the window
    // gpulife->show();
  }
}

int main(int argc, char** argv) {
  GPULife * gpuLife = new GPULife(128, 360);
  drive(gpuLife);
  return 0;
}
