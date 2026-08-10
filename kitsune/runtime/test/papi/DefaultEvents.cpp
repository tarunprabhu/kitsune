// REQUIRES: kitsune-papi
//
// Check some PAPI event names that may be passed to __kitpapi_start. This
// should be exhaustive, but that is pretty tedious, and it is not clear what
// advantage there is to doing that. We rely on grey-box testing anyway, so this
// is a best-effort test.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Event 'PAPI_{{.+}}' added to epoch 'hadji'
// CHECK: Event 'PAPI_{{.+}}' not available

#include "TestHelpers.h"
#include "papi/kitpapi.h"

#include "papi.h"

#include <ctype.h>
#include <string.h>

CTOR(RT_PAPI)

#define MAX_EVENTS 100

typedef struct {
  int evt;
  char *name;
  bool avail;
} Event;

static void getEvent(int evt, Event *e) {
  e->evt = evt;
  e->name = NULL;
  e->avail = false;

  PAPI_event_info_t evtInfo;
  if (PAPI_get_event_info(evt, &evtInfo) != PAPI_OK)
    return;

  e->name = strdup(&evtInfo.symbol[5]);
  for (unsigned i = 0; e->name[i] != '\0'; ++i)
    e->name[i] = tolower(e->name[i]);
  e->avail = evtInfo.count;
}

int main(int argc, char *argv[]) {
  // It is not clear what or'ing with 0 is meant to signify. This is how it is
  // implemented in papi_avail.c.
  int evt = PAPI_PRESET_MASK | 0;
  if (PAPI_enum_event(&evt, PAPI_ENUM_FIRST) != PAPI_OK)
    return 0;

  unsigned i = 0;
  Event evts[MAX_EVENTS];

  getEvent(evt, &evts[i++]);
  while (PAPI_enum_event(&evt, PAPI_ENUM_EVENTS) == PAPI_OK && i < MAX_EVENTS)
    getEvent(evt, &evts[i++]);

  const char *evtAvail = NULL;
  const char *evtNotAvail = NULL;
  for (unsigned i = 0; i < MAX_EVENTS; ++i) {
    if (!evts[i].name)
      continue;
    else if (evts[i].avail && !evtAvail)
      evtAvail = evts[i].name;
    else if (!evts[i].avail && !evtNotAvail)
      evtNotAvail = evts[i].name;
  }

  KitPAPIEpoch *e =
      __kitpapi_start("hadji", /*thread=*/0, 2, evtAvail, evtNotAvail);
  __kitpapi_stop(e);

  return 0;
}
