#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

#include <stdio.h>
#include <stdlib.h>

#include "gpulife.hpp"

void redraw(GPULife* gpulife, Display* dis, GC gc, int win){
  static int generation = 0;
  auto dims = gpulife->dims();
  const auto *cells = gpulife->cells();

  XFlush(dis);
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

  static GC textGC;
  static bool inited;
  if (!inited) {
    XGCValues grvals;
    memset(&grvals, 0, sizeof(grvals));
    auto fontInfo = XLoadQueryFont(dis, "fixed");
    grvals.font = fontInfo->fid;
    grvals.background = BlackPixel(dis, XDefaultScreen(dis));
    grvals.foreground = WhitePixel(dis, XDefaultScreen(dis));
    textGC = XCreateGC(dis, win, GCFont|GCForeground|GCBackground, &grvals);
  }
  char msg [20];
  snprintf(msg, 20, "gen %05d\n", generation++);
  XDrawString(dis, win, textGC, 10, 50, msg, strlen(msg));
  XSync(dis, False);
}

void handleEvent(GPULife* g, const XEvent* ev){
}

void drive(GPULife* gpulife){
  auto dims = gpulife->dims();

  Display *dis = XOpenDisplay(NULL);
  int screen = XDefaultScreen(dis);

  Window win = XCreateSimpleWindow(dis, DefaultRootWindow(dis),
        0, 0,
        dims.x, dims.y, 5,
        BlackPixel(dis, screen), WhitePixel(dis, screen));
  XSetStandardProperties(dis, win, "GPULife", "Oy!", None, NULL, 0, NULL);
  XSelectInput(dis, win,         
        ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
        ButtonPressMask | ButtonReleaseMask  | StructureNotifyMask );

  XMapWindow(dis, win);
  auto gc = DefaultGC(dis, screen);
  XMapRaised(dis, win);

  // auto pixmap = XCreatePixmap(dis, win, dims.x, dims.y, 1);

  auto x11fd = ConnectionNumber(dis);
  for (auto i = 0;; i++) {
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

    if ((i % 10) == 0) {
      redraw(gpulife, dis, gc, win);
    }
    // XXX put it in the window
    // gpulife->show();
  }
}

int main(int argc, char** argv) {
  srand(getpid());
  auto X = 360;
  auto Y = 128;
  if (argc >= 3) {
    X = atoi(argv[1]);
    Y = atoi(argv[2]);
  }
  GPULife * gpuLife = new GPULife(X, Y);
  drive(gpuLife);
  return 0;
}
