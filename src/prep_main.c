#include "prep_search.h"
#include <string.h>
int prep_run_command(int argc, char **argv);
int main(int argc, char **argv)
{
    if (argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        prep_print_help();
        return 0;
    }
    return prep_run_command(argc, argv);
}
