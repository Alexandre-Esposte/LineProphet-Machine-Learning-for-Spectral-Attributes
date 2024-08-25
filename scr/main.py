from linegenerator.database_generator import database_generator


database_generator(60_000,'train')
database_generator(500,'test')

print('Done')